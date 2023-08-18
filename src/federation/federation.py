# -*- coding: utf-8 -*-
"""
Created on Feb 1, 2022

.. codeauthor:: L. Calvo-Bartolomé (lcalvo@pa.uc3m.es)
"""


import threading

from src.federation.federation_client import FederationClient


class Federation:
    """Class to describe the set of clients that compose a federation.
    """

    def __init__(self, logger=None):
        self.federation_lock = threading.RLock()
        self.federation = {}
        self.federation_clients = []

        if logger:
            self.logger = logger
        else:
            import logging
            logging.basicConfig(level='INFO')
            self.logger = logging.getLogger('Federation')

    def connect_consensus(self, client):
        """
        Class to register the connection of a client in the federation for the first time, i.e. for the agreement of a common vocabulary. 

        As the client gets registered into the federation when it sends a GRPC message to the server, a counter is kept in order to account for the number times a client has tried to communicate with it. To do so, we save each client identification obtained from the context as the key of a dictionary, incrementing by one its corresponding value at each time the client connects with the server. Additionally, an object the class FederationClient is added to the list clients in the federation at each time a new client connects to it.

        Parameters
        ----------
        client : str
            Client idenfication obtained from the context that provides information on the RPC
        """

        self.logger.info("Client {} connecting for consensus".format(client))
        with self.federation_lock:
            if client not in self.federation:
                self.federation[client] = 1
                new_federation_client = \
                    FederationClient(federation_key=client)
                self.federation_clients.append(new_federation_client)
            else:
                self.federation[client] += 1

    def connect_update(self, client, gradients, current_mb, current_epoch, current_id_msg, max_iter):
        """
        Class to register the connection of a client in the federation for an update. 

        Parameters
        ----------
        client: str
            Client idenfication obtained from the context that provides information on the RPC
        gradients: List[List[str,Pytorch.Tensor]]
            Gradient that the client is sending to the server at "current_iter" on "current_id_msg"
        current_mb: int
            Epoch that corresponds with the gradient that is being sent by the client
        current_id_msg: int
            Id of the message with which the gradient is being sent
        max_iter: int
            Number of epochs with which the model is being trained
        """

        self.logger.info("Client {} connecting for update".format(client))
        with self.federation_lock:
            if client not in self.federation:
                self.federation[client] = 1
            else:
                self.federation[client] += 1
            id_client = FederationClient.get_pos_by_key(
                client, self.federation_clients)
            if id_client != -1:
                print(type(current_epoch))
                self.federation_clients[id_client].set_num_max_iter(max_iter)
                self.federation_clients[id_client].update_client_state(
                    gradients, current_mb, current_epoch, current_id_msg)

    def connect_waiting_or_consensus(self, client, waiting):
        """
        Class to register the connection of a client in the federation when it is waiting until all the clients in the federation have sent their local corpus so the server can send the consensed vocabulary, or when the server is sending the consensed vocabulary

        Parameters
        ----------
        client: str
            Client idenfication obtained from the context that provides information on the RPC
        waiting: bool
            Whether the client is waiting for the vocabulary consensus sending or the vocabulary consensus is already happening
        """

        with self.federation_lock:
            if client not in self.federation:
                self.federation[client] = 1
            else:
                self.federation[client] += 1
        if waiting:
            self.logger.info("Client {} connecting waiting for training to start".format(client))
        else:
            self.logger.info(
                "Client {} connected for vocabulary consensus".format(client))

    def disconnect(self, client):
        """
        Method to unregister a client from the federation. 

        As the registration of the user in the federation is described by a counter that relates to the number of times the user has sent a GRPC message to the server, the client is not completely unregister from the federation until this counter is set to 0. At this time, the client is also removed from the list FederationClient objects that describe the set clients that are connected in the federationl.

        Parameters
        ----------
        client : str
            Client idenfication obtained from the context that provides information on the RPC

        Raises
        -------
        RuntimeError
            A runtime error is raised when it is attempted to remove a client that has not previously connected to the federation.
        """

        self.logger.info("Client {} disconnecting".format(client))
        with self.federation_lock:
            if client not in self.federation:
                raise RuntimeError(
                    "Tried to disconnect client '{}' but it was never connected.".format(client))
            self.federation[client] -= 1
            if self.federation[client] == 0:
                del self.federation[client]
                client_to_remove = FederationClient.get_pos_by_key(
                    client, self.federation_clients)
                if self.federation_clients[client_to_remove].current_epoch == self.federation_clients[client_to_remove].current_id_msg:
                    del self.federation_clients[client_to_remove]

    def getClients(self):
        """
        Get keys of the clients that are connected to the federation.

        Returns
        -------
        clients_info:
            Client idenfications from all the clients connected to the federation obtained from the context that provides information on the RPC
        """

        with self.federation_lock:
            clients_info = self.federation.keys()
            return clients_info
