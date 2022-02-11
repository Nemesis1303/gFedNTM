# -*- coding: utf-8 -*-
"""
@author: lcalv
******************************************************************************
***                       CLASS FEDERATION CLIENT                          ***
******************************************************************************
"""
##############################################################################
#                                IMPORTS                                     #
##############################################################################


class FederationClient:
    """Class for representing a client that has connected to the server for the federated learning of 
       a CTM or ProdLDA model.
    """

    def __init__(self, federation_key, path_tmp_local_corpus):
        self.federation_key = federation_key
        self.path_tmp_local_corpus = path_tmp_local_corpus
        self.tensor = None
        self.current_iter = None
        self.current_id_msg = None
        self.num_max_iter = None
        self.vocab_sent = False
        
    def set_num_max_iter(self, num_max_iter):
        self.num_max_iter = num_max_iter

    def update_client_state(self, tensor, current_iter, current_id_msg):
        self.tensor = tensor
        self.current_iter = current_iter,
        self.current_id_msg = current_id_msg

    def get_pos_by_key(key, federation_clients):
        """It searchs a client with the specified key over the given list of FederationClient objects
           and when found, it returns its position in the list.
        Args:
            * key (int): Dictionary id of the client 
            * federation_clients (FederationClient): List of FederationClient objects representing the clients
                                                   that are connected to the client.

        Returns:
            * int: Position on the list of the searched client
        """
        for client_pos in range(len(federation_clients)):
            if key == federation_clients[client_pos].federation_key:
                return client_pos
            else:
                print("No client with specified key was found")
                return -1
