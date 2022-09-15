# -*- coding: utf-8 -*-
"""
@author: lcalv
******************************************************************************
***                        AUXILIARY FUNCTIONS                             ***
******************************************************************************
"""
import numpy as np
import pickle
from scipy import sparse
import json
from federation import federated_pb2

##############################################################################
#                                CONSTANTS                                   #
##############################################################################
CHUNK_SIZE = 1024 * 1024  # 1MB


def get_type_from_string(str_dtype):
    """Gets the dtype object from its string characterization"""
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


def unpickler(file: str):
    """Unpickle file"""
    with open(file, 'rb') as f:
        return pickle.load(f)


def pickler(file: str, ob):
    """Pickle object to file"""
    with open(file, 'wb') as f:
        pickle.dump(ob, f)
    return

def save_model_as_npz(npzfile, client):
    """Saves the matrixes that characterize a topic model in a numpy npz filel.

    Args:
        npzfile (str): Name of the file in which the model will be saved
        client (): 
    """

    if isinstance(client.local_model.thetas, sparse.csr_matrix):
        np.savez(
            npzfile,
            betas=client.local_model.betas,
            thetas_data=client.local_model.thetas.data,
            thetas_indices=client.local_model.thetas.indices,
            thetas_indptr=client.local_model.thetas.indptr,
            thetas_shape=client.local_model.thetas.shape,
            ntopics=client.local_model.n_components,
            topics=client.local_model.topics
        )
    else:
        np.savez(
            npzfile,
            betas=client.local_model.betas,
            thetas=client.local_model.doc_topic_distrib,
            ntopics=client.local_model.n_components,
            topics=client.local_model.topics
        )
