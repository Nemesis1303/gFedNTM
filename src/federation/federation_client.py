# -*- coding: utf-8 -*-
"""
Created on Feb 1, 2022

.. codeauthor:: L. Calvo-Bartolomé (lcalvo@pa.uc3m.es)
"""


class FederationClient:
    """Class for representing a client that has connected to the server for the federated learning of a topic model.
    """

    def __init__(self, federation_key):

        self.federation_key = federation_key
        self.client_id = None
        self.tensors = None
        self.current_epoch = -1
        self.current_id_msg = -2
        self.num_max_iter = -3
        self.current_mb = -4
        self.nr_samples = -5
        self.vocab_sent = False
        self.can_get_update = False
        self.global_epoch = -3
        self.ready_for_training = False

    def set_num_max_iter(self, num_max_iter):
        """
        Sets the maximum number of iterations of the training process associated with an specific client.
        """

        self.num_max_iter = num_max_iter

    def update_client_state(self, tensors, current_mb, current_epoch, current_id_msg):
        """Sets the state of the client, that is:
        - current tensor that is being sent to the server
        - curent minibatch to which the sent tensor belongs to
        - current epoch to which the minibatch belongs to
        - id of the message in which the current tensor is being sent
        """

        self.tensors = tensors
        self.current_mb = current_mb
        self.current_epoch = current_epoch,
        self.current_id_msg = current_id_msg

    def get_pos_by_key(key, federation_clients):
        """
        It searchs a client with the specified key over the given list of FederationClient objects and when found, it returns its position in the list.

        Parameters
        ----------
        key : int
            Dictionary id of the client
        federation_clients: FederationClient
            List of FederationClient objects representing the clients that are connected to the client.

        Returns
        -------
        client_pos: int
            Position on the list of the searched client or -1 when no client with the specified key was found
        """

        for client_pos in range(len(federation_clients)):
            if key == federation_clients[client_pos].federation_key:
                return client_pos
            # else:
            #    print("No client with specified key was found")
        return -1

    def set_can_get_update_by_key(key, federation_clients, update):
        """It searches for the client described by the given key in the list of federation_clients, and when found, the boolean update specifies whether such client can receive an update or not.
        """
        for client_pos in range(len(federation_clients)):
            if key == federation_clients[client_pos].federation_key:
                federation_clients[client_pos].can_get_update = update
    
    def set_can_start_training(key, federation_clients, update):
        """It searches for the client described by the given key in the list of federation_clients, and when found, the boolean update specifies whether the client can start the training or not.
        """
        for client_pos in range(len(federation_clients)):
            if key == federation_clients[client_pos].federation_key:
                federation_clients[client_pos].ready_for_training = update

    def set_global_epoch_by_key(key, federation_clients, global_epoch):
        """_summary_

        :param key: _description_
        :type key: _type_
        :param federation_clients: _description_
        :type federation_clients: _type_
        :param global_epoch: _description_
        :type global_epoch: _type_
        """
        for client_pos in range(len(federation_clients)):
            if key == federation_clients[client_pos].federation_key:
                federation_clients[client_pos].global_epoch = global_epoch

    def set_id_by_key(key, federation_clients, id):
        """Sets the ID of a client with the specified key in the list of federation_clients.
        """
        for client_pos in range(len(federation_clients)):
            if key == federation_clients[client_pos].federation_key:
                federation_clients[client_pos].client_id = id