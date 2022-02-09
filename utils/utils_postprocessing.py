# -*- coding: utf-8 -*-
"""
@author: lcalv
******************************************************************************
***                           UTILS POSTPROCESSING                         ***
******************************************************************************
"""
##############################################################################
#                                IMPORTS                                     #
##############################################################################
import numpy as np


def convert_topic_word_to_init_size(vocab_size, model, model_type,
                                    ntopics, id2token, all_words):
    """It converts the topic-word distribution matrix obtained from the training of a model into a matrix with the dimensions of the original topic-word distribution, assigning zeros to those words that are not present in the corpus. 
    It is only of use in case we are training a model over a synthetic dataset, so as to later compare the performance of the attained model in what regards to the similarity between the original and the trained model.

    Args:
        * vocab_size (int): Size of the synethic'data vocabulary.
        * model (AVITM): Model whose topic-word matrix is being transformed.
        * model_type (str): Type of the trained model (e.g. AVITM)
        * ntopics (int): Number of topics of the trained model.
        * id2token (List[tuple]): Mappings with the content of the document-term matrix.
        * all_words (List[str]): List of all the words of the vocabulary of size vocab_size.

    Returns:
        * ndarray: Normalized transormed topic-word distribution.
    """
    if model_type == "avitm":
        w_t_distrib = np.zeros((10, vocab_size), dtype=np.float64)
        wd = model.get_topic_word_distribution()
        for i in np.arange(ntopics):
            for idx, word in id2token.items():
                for j in np.arange(len(all_words)):
                    if all_words[j] == word:
                        w_t_distrib[i, j] = wd[i][idx]
                        break
        sum_of_rows = w_t_distrib.sum(axis=1)
        normalized_array = w_t_distrib / sum_of_rows[:, np.newaxis]
        return normalized_array
    else:
        print("Method not impleemnted for the selected model type")
        return None
