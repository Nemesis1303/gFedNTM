# -*- coding: utf-8 -*-
"""
@author: lcalv
******************************************************************************
***                         CLASS FEDERATION                               ***
******************************************************************************
"""
##############################################################################
#                                IMPORTS                                     #
##############################################################################
import threading

from federation.federation_client import FederationClient


class Federation:
    """Class to describe the set  clients that compose a federation.
    """

    def __init__(self):
        self.federation_lock = threading.RLock()
        self.federation = {}
        self.federation_clients = []

    def connect(self, client, id_client, gradient, current_iter, current_id_msg, max_iter):
        """Class to register the connection  a client in the federation. As the client gets registered into the federation when it sends a GRPC message to the server, a counter is kept in order to account for the number  times a client has tried to communicate with the server. To do so, we save each client identification obtained from the context as the key  a dictionary, incrementing by one its corresponding value at each time the client connects with the server. Additionally, an object the class FederationClient is added to the list clients in the federation at each time a new client connects to it.

        Args:
            * client (str): Client idenfication obtained from the context that provides information on the RPC
            * id_client (int): Id  the client.
            * gradient (Pytorch.Tensor): Gradient that the client is sending to the server at "current_iter" on "current_id_msg"
            * current_iter (int): Iteration that corresponds with the gradient that is being sent by the client.
            * current_id_msg (int): Id  the message at which the gradient is being sent.
            * max_iter (int): Number of epochs with which the model is being trained.
        """
        print("Client {} connecting".format(client))
        with self.federation_lock:
            if client not in self.federation:
                self.federation[client] = 1
                new_federation_client = FederationClient(id=id_client,
                                                         federation_key=client,
                                                         tensor=gradient, current_iter=current_iter,
                                                         current_id_msg=current_id_msg,
                                                         num_max_iter=max_iter)
                self.federation_clients.append(new_federation_client)
            else:
                self.federation[client] += 1

    def connect_waiting(self, client):
        print("Client {} connecting waiting".format(client))
        with self.federation_lock:
            if client not in self.federation:
                self.federation[client] = 1
            else:
                self.federation[client] += 1

    def disconnect(self, client):
        """Class to unregister a client from the federation. As the registration  the user in the federation is described by a counter that relates to the number  times the user has sent a GRPC message to the server, the client is not completely unregister from the federation until this counter is set to 0. At this time, the client is also removed from the list  FederationClient objects that describe the set  clients that are connected in the federationl.

        Args:
            * client (str): Client idenfication obtained from the context that provides information on the RPC

        Raises:
            * RuntimeError: A runtime error is raised when it is attempted to remove a client that has not previously connected to the federation.
        """
        print("Client {} disconnecting".format(client))
        with self.federation_lock:
            if client not in self.federation:
                raise RuntimeError(
                    "Tried to disconnect client '{}' but it was never connected.".format(client))
            self.federation[client] -= 1
            if self.federation[client] == 0:
                del self.federation[client]
                client_to_remove = FederationClient.get_pos_by_key(
                    client, self.federation_clients)
                if self.federation_clients[client_to_remove].current_iter == self.federation_clients[client_to_remove].current_id_msg:
                    del self.federation_clients[client_to_remove]

    def getClients(self):
        """Get keys of the clients that are connected to the federation.

        Returns:
            [type]: Client idenfications from all the clients connected to the federation obtained from the context that provides information on the RPC
        """
        with self.federation_lock:
            return self.federation.keys()
