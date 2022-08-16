#!/usr/bin/env python
# coding: utf-8

# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Set-up-of-the-notebook" data-toc-modified-id="Set-up-of-the-notebook-1">Set up of the notebook</a></span></li><li><span><a href="#Generation-of-Synthetic-data" data-toc-modified-id="Generation-of-Synthetic-data-2">Generation of Synthetic data</a></span></li><li><span><a href="#Training-and-Evaluating-model-for-centralized-approach" data-toc-modified-id="Training-and-Evaluating-model-for-centralized-approach-3">Training and Evaluating model for centralized approach</a></span><ul class="toc-item"><li><ul class="toc-item"><li><span><a href="#Topic-evaluation" data-toc-modified-id="Topic-evaluation-3.0.1">Topic evaluation</a></span></li><li><span><a href="#Doc-similarity-evaluation" data-toc-modified-id="Doc-similarity-evaluation-3.0.2">Doc similarity evaluation</a></span></li></ul></li></ul></li><li><span><a href="#Training-and-Evaluating-model-for-isolated-node-0" data-toc-modified-id="Training-and-Evaluating-model-for-isolated-node-0-4">Training and Evaluating model for isolated node 0</a></span><ul class="toc-item"><li><ul class="toc-item"><li><span><a href="#Topic-evaluation" data-toc-modified-id="Topic-evaluation-4.0.1">Topic evaluation</a></span></li><li><span><a href="#Doc-similarity-evaluation" data-toc-modified-id="Doc-similarity-evaluation-4.0.2">Doc similarity evaluation</a></span></li></ul></li></ul></li></ul></div>

# # Set up of the notebook

# In[167]:


# Common imports 
import numpy as np
import pandas as pd
import zipfile as zp
from pathlib import Path
from gensim.utils import check_output
from sklearn.preprocessing import normalize


# In[168]:


mallet_path = Path('./My\ Drive/github/IntelComp/WP4/topicmodeler/mallet-2.0.8/bin/mallet')


# # Generation of Synthetic data

# In[186]:


def rotateArray(arr, n, d):
    temp = []
    i = 0
    while (i < d):
        temp.append(arr[i])
        i = i + 1
    i = 0
    while (d < n):
        arr[i] = arr[d]
        i = i + 1
        d = d + 1
    arr[:] = arr[: i] + temp
    return arr

# Topic modeling settings
vocab_size = 5000
n_topics = 50
beta = 1e-2
alpha = 1/n_topics
n_docs = 1000
nwords = (150, 250) #Min and max lengths of the documents

# Nodes settings
n_nodes = 5
frozen_topics = 5
prior_frozen = frozen_topics * [alpha]
own_topics = int((n_topics-frozen_topics)/n_nodes)
prior_nofrozen = own_topics * [alpha] + (n_topics-frozen_topics-own_topics) * [alpha/10000]
#print(prior_frozen + prior_nofrozen)


# In[187]:


# Step 1 - generation of topics
topic_vectors = np.random.dirichlet(vocab_size*[beta], n_topics)
print('Probabilidades ordenadas para el primer vector de tópicos:')
print(np.sort(topic_vectors[0])[::-1])
print(topic_vectors.shape)


# In[188]:


#Here we compare alignment of the topic_vector matrix with itself and with another randomly generated matrix
print('Tópicos (equivalentes) identificados correctamente (true):', np.sum(np.max(np.sqrt(topic_vectors).dot(np.sqrt(topic_vectors.T)), axis=0)))
topic_vectors2 = np.random.dirichlet(vocab_size*[beta], n_topics)
print('Tópicos (equivalentes) identificados correctamente (random):', np.sum(np.max(np.sqrt(topic_vectors2).dot(np.sqrt(topic_vectors.T)), axis=0)))


# In[189]:


# Step 2 - generation of document topic proportions
doc_topics_all = []
for i in np.arange(n_nodes):
    doc_topics = np.random.dirichlet(prior_frozen + prior_nofrozen, n_docs)
    prior_nofrozen = rotateArray(prior_nofrozen, len(prior_nofrozen), own_topics)
    doc_topics_all.append(doc_topics)


# In[190]:


# Step 3 - Document generation
documents_all = []
z_all = []

for i in np.arange(n_nodes):
    documents = [] # Document words
    #z = [] # Assignments
    for docid in np.arange(n_docs):
        doc_len = np.random.randint(low=nwords[0], high=nwords[1])
        this_doc_words = []
        #this_doc_assigns = []
        for wd_idx in np.arange(doc_len):
            tpc = np.nonzero(np.random.multinomial(1, doc_topics_all[i][docid]))[0][0]
            #this_doc_assigns.append(tpc)
            word = np.nonzero(np.random.multinomial(1, topic_vectors[tpc]))[0][0]
            this_doc_words.append('wd'+str(word))
        #z.append(this_doc_assigns)
        documents.append(this_doc_words)
    documents_all.append(documents)
    #z_all.append(z)


# # Training and Evaluating model for centralized approach

# In[191]:


my_corpus = [doc for docs_node in documents_all for doc in docs_node]

with open('./models_aux/corpus.txt', 'w') as fout:
    [fout.write(str(idx) + ' 0 ' + ' '.join(doc) + '\n') for idx,doc in enumerate(my_corpus)]


# In[88]:


cmd = mallet_path.as_posix() +     ' import-file --preserve-case --keep-sequence ' +     '--remove-stopwords --token-regex "[\p{L}\p{N}][\p{L}\p{N}\p{P}]*"' +     ' --input ./models_aux/corpus.txt --output ./models_aux/corpus.mallet'
print(cmd)


# In[155]:


with open('./models_aux/train.config', 'w', encoding='utf8') as fout:
    fout.write('input = ../models_aux/corpus.mallet\n')
    fout.write('num-topics = ' + str(n_topics) + '\n')
    fout.write('alpha = 1\n')
    fout.write('optimize-interval = 10\n')
    fout.write('num-threads = 4\n')
    fout.write('num-iterations = 1000\n')
    fout.write('doc-topics-threshold = 0\n')
    fout.write('output-doc-topics = ../models_aux/doc-topics.txt\n')
    fout.write('word-topic-counts-file = ../models_aux/word-topic-counts.txt\n')
    fout.write('output-topic-keys = ../models_aux/topickeys.txt\n')
cmd = mallet_path.as_posix() + ' train-topics --config ./models_aux/train.config'
print(cmd)


# ### Topic evaluation
# 
# 

# In[192]:


#Recover and build beta matrix
beta = np.zeros((n_topics, vocab_size))

with open('./models_aux/word-topic-counts.txt', 'r', encoding='utf8') as fin:
    for line in fin.readlines():
        tokens = line.split()[1:]
        pos = int(tokens[0][2:])
        for el in tokens[1:]:
            tpc = int(el.split(':')[0])
            cnt = int(el.split(':')[1])
            beta[tpc,pos] = cnt

beta = normalize(beta,axis=1,norm='l1')


# In[193]:


print('Tópicos (equivalentes) evaluados correctamente:', np.sum(np.max(np.sqrt(beta).dot(np.sqrt(topic_vectors.T)), axis=0)))


# ### Doc similarity evaluation

# In[136]:


sim_mat_theoretical = np.sqrt(doc_topics_all[0]).dot(np.sqrt(doc_topics_all[0].T))


# In[139]:


thetas = np.loadtxt('models_aux/doc-topics.txt', delimiter='\t', dtype=np.float32)[:,2:][:n_docs,:]
thetas[thetas<3e-3] = 0
thetas = normalize(thetas,axis=1,norm='l1')

sim_mat_actual = np.sqrt(thetas).dot(np.sqrt(thetas.T))


# In[140]:


print('Difference in evaluation of doc similarity:', np.sum(np.abs(sim_mat_theoretical - sim_mat_actual))/n_docs)


# # Training and Evaluating model for isolated node 0
# 
# ---
# 
# 

# In[194]:


my_corpus = documents_all[0]

with open('./models_aux/corpus.txt', 'w') as fout:
    [fout.write(str(idx) + ' 0 ' + ' '.join(doc) + '\n') for idx,doc in enumerate(my_corpus)]


# In[88]:


cmd = mallet_path.as_posix() +     ' import-file --preserve-case --keep-sequence ' +     '--remove-stopwords --token-regex "[\p{L}\p{N}][\p{L}\p{N}\p{P}]*"' +     ' --input ./models_aux/corpus.txt --output ./models_aux/corpus.mallet'
print(cmd)


# In[87]:


with open('./models_aux/train.config', 'w', encoding='utf8') as fout:
    fout.write('input = ../models_aux/corpus.mallet\n')
    fout.write('num-topics = ' + str(n_topics) + '\n')
    fout.write('alpha = 5\n')
    fout.write('optimize-interval = 10\n')
    fout.write('num-threads = 4\n')
    fout.write('num-iterations = 1000\n')
    fout.write('doc-topics-threshold = 0\n')
    fout.write('output-doc-topics = ../models_aux/doc-topics.txt\n')
    fout.write('word-topic-counts-file = ../models_aux/word-topic-counts.txt\n')
    fout.write('output-topic-keys = ../models_aux/topickeys.txt\n')
cmd = mallet_path.as_posix() + ' train-topics --config ./models_aux/train.config'
print(cmd)


# ### Topic evaluation
# 
# 

# In[195]:


#Recover and build beta matrix
beta = np.zeros((n_topics, vocab_size))

with open('./models_aux/word-topic-counts.txt', 'r', encoding='utf8') as fin:
    for line in fin.readlines():
        tokens = line.split()[1:]
        pos = int(tokens[0][2:])
        for el in tokens[1:]:
            tpc = int(el.split(':')[0])
            cnt = int(el.split(':')[1])
            beta[tpc,pos] = cnt

beta = normalize(beta,axis=1,norm='l1')


# In[196]:


print('Tópicos (equivalentes) evaluados correctamente:', np.sum(np.max(np.sqrt(beta).dot(np.sqrt(topic_vectors.T)), axis=0)))


# ### Doc similarity evaluation

# In[144]:


sim_mat_theoretical = np.sqrt(doc_topics_all[0]).dot(np.sqrt(doc_topics_all[0].T))


# In[147]:


thetas = np.loadtxt('models_aux/doc-topics.txt', delimiter='\t', dtype=np.float32)[:,2:][:n_docs,:]
thetas[thetas<3e-3] = 0
thetas = normalize(thetas,axis=1,norm='l1')

sim_mat_actual = np.sqrt(thetas).dot(np.sqrt(thetas.T))


# In[148]:


print('Difference in evaluation of doc similarity:', np.sum(np.abs(sim_mat_theoretical - sim_mat_actual))/n_docs)


# In[ ]:




