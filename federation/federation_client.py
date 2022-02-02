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

    def __init__(self, id, federation_key, tensor, current_iter, current_id_msg):
        self.id = id
        self.federation_key = federation_key
        self.tensor = tensor
        self.current_iter = current_iter
        self.current_id_msg = current_id_msg

    def get_pos_by_id(id, federation_clients):
        """It searchs a client with the specified id over the given list of FederationClient objects
           and when found, it returns its position in the list.
        Args:
            * id (int): Dictionary id of the client 
            * federation_clients (FederationClient): List of FederationClient objects representing the clients
                                                   that are connected to the client.

        Returns:
            * int: Position on the list of the searched client
        """
        for client_pos in range(len(federation_clients)):
            if id == federation_clients[client_pos].id:
                return client_pos
            else:
                print("No client with specified key was found")
                return -1

    def update_client_state(self, tensor, current_iter, current_id_msg):
        self.tensor = tensor
        self.current_iter = current_iter, 
        self.current_id_msg = current_id_msg