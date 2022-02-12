# -*- coding: utf-8 -*-
"""
@author: lcalv
******************************************************************************
***                        AUXILIARY FUNCTIONS                             ***
******************************************************************************
"""
##############################################################################
#                                IMPORTS                                     #
##############################################################################
import numpy as np
import pickle

from federation import federated_pb2

##############################################################################
#                                CONSTANTS                                   #
##############################################################################
CHUNK_SIZE = 1024 * 1024  # 1MB


def get_type_from_string(str_dtype):
    # @ TODO: Make match-case when Pytorch works with Python 3.10
    print(str_dtype)
    if str_dtype == "float32":
        dytpe = np.float32
    elif str_dtype == "float64":
        dytpe = np.float64
    return dytpe


def get_file_chunks(filename):
    with open(filename, 'rb') as f:
        while True:
            piece = f.read(CHUNK_SIZE)
            if len(piece) == 0:
                return
            yield federated_pb2.Chunk(buffer=piece)


def save_chunks_to_file(chunks, filename):
    with open(filename, 'wb') as f:
        for chunk in chunks:
            f.write(chunk.buffer)


def save_in_pickle(structure, pickle_to_save_in):
    with open(pickle_to_save_in, 'wb') as f:
        pickle.dump(structure, f)


def save_corpus_in_file(corpus, file):
    documents = [' '.join(el) for el in corpus]
    with open(file, 'w') as fout:
        for idx, doc in enumerate(documents):
            fout.write(doc + '\n')


def get_corpus_from_file(file):
    corpus = []
    with open(file, 'r') as f:
        for l in f.readlines():
            corpus.append(l.split())
    return corpus
