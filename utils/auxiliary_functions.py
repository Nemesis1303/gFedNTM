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
