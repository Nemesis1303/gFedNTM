import numpy as np

def rotateArray(arr: np.ndarray, n: int, d: int) -> np.ndarray:
    """Rotates an array of size n by d elements. 
    
    Parameters
    ----------
    arr: np.ndarray
        Array to rotate
    n: int
        Size of the array
    d: int 
        Number of positions to rotate the array
    
    Returns
    --------
    np.ndarray: The rotated array
    """
    
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

# Step 1 - generation of topics
topic_vectors = np.random.dirichlet(vocab_size*[beta], n_topics)
print('Probabilidades ordenadas para el primer vector de tópicos:')
print(np.sort(topic_vectors[0])[::-1])
print(topic_vectors.shape)

# Step 2 - generation of document topic proportions
doc_topics_all_gt = []
for i in np.arange(n_nodes):
    doc_topics = np.random.dirichlet(prior_frozen + prior_nofrozen, n_docs)
    prior_nofrozen = rotateArray(prior_nofrozen, len(prior_nofrozen), own_topics)
    doc_topics_all_gt.append(doc_topics)

# Step 3 - Document generation
documents_all = []
z_all = []

for i in np.arange(n_nodes):
  documents = [] # Document words
  z = [] # Assignments
  for docid in np.arange(n_docs):
      doc_len = np.random.randint(low=nwords[0], high=nwords[1])
      this_doc_words = []
      #this_doc_assigns = []
      for wd_idx in np.arange(doc_len):
          tpc = np.nonzero(np.random.multinomial(1, doc_topics_all_gt[i][docid]))[0][0]
          #this_doc_assigns.append(tpc)
          word = np.nonzero(np.random.multinomial(1, topic_vectors[tpc]))[0][0]
          this_doc_words.append('wd'+str(word))
      #z.append(this_doc_assigns)
      documents.append(this_doc_words)
  print("Documents of node", str(i), "generated.")
  documents_all.append(documents)
  #z_all.append(z)

  documents = [' '.join(el) for el in documents]
  name_corpus_txt = "synthetic_corpus_node" + str(i) + ".txt"
  name_corpus_npz = "synthetic_corpus_node" + str(i) + ".npz"
  with open(name_corpus_txt, 'w') as fout:
    for idx, doc in enumerate(documents):
        fout.write(str(idx) + ' 0 ' + doc + '\n')
  np.savez(name_corpus_npz, vocab_size=vocab_size, n_topics=n_topics, beta=beta, alpha=alpha,
        n_docs=n_docs, nwords=nwords, topic_vectors=topic_vectors, doc_topics=doc_topics,
        documents=documents, z=z)
  

np.savez('synthetic_all_nodes.npz', n_nodes = n_nodes, vocab_size=vocab_size, n_topics=n_topics, frozen_topics = frozen_topics, beta=beta, alpha=alpha, n_docs=n_docs, nwords=nwords, topic_vectors=topic_vectors, doc_topics=doc_topics_all_gt,
        documents=documents_all)