# -*- coding: utf-8 -*-
"""
@author: lcalv
******************************************************************************
***                                UTILS                                   ***
******************************************************************************
"""
##############################################################################
#                                IMPORTS                                     #
##############################################################################
import numpy as np


def to_bow(data, min_length):
    """Convert index lists to bag of words representation of documents.
    From "to_bow" method at:
    https://github.com/estebandito22/PyTorchAVITM/blob/master/train.py
    """

    vect = [np.bincount(x[x != np.array(None)].astype('int'), minlength=min_length)
            for x in data if np.sum(x[x != np.array(None)]) != 0]
    return np.array(vect)

def preprocess_data_avitm(n_nodes, corpus_nodes):
    train_datasets = []
    id2tokens 



avitms = []
id2tokens = []
for corpus_node in documents_all:
  cv = CountVectorizer(input='content', lowercase=True, stop_words='english',
                      max_df=0.99, min_df=0.01, binary=False)
  
  docs = [" ".join(corpus_node[i]) for i in np.arange(len(corpus_node))]

  train_bow = cv.fit_transform(docs)
  train_bow = train_bow.toarray()

  idx2token = cv.get_feature_names()
  input_size = len(idx2token)

  id2token = {k: v for k, v in zip(range(0, len(idx2token)), idx2token)}
  id2tokens.append(id2token)

  train_data = BOWDataset(train_bow, idx2token)

  avitm = AVITM(input_size=input_size, n_components=10, model_type='prodLDA',
                hidden_sizes=(100, 100), activation='softplus', dropout=0.2,
                learn_priors=True, batch_size=64, lr=2e-3, momentum=0.99,
                solver='adam', num_epochs=100, reduce_on_plateau=False)

  avitm.fit(train_data)
  avitms.append(avitm)

  train_datasets.append(train_data)