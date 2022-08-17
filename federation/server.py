# -*- coding: utf-8 -*-
"""
@author: lcalv
******************************************************************************
***                      CLASS FEDERATED SERVER                            ***
******************************************************************************
"""
import os
import time

import numpy as np
from gensim.test.utils import get_tmpfile
from utils.auxiliary_functions import (get_file_chunks, get_type_from_string,
                                       save_chunks_to_file,
                                       save_corpus_in_file)
from waiting import wait

from federation import (federated_pb2, federated_pb2_grpc, federation,
                        federation_client)

GRPC_TRACE = all


class FederatedServer(federated_pb2_grpc.FederationServicer):
    """Class that describes the behaviour of the GRPC server to which several clients are connected to create a federation for the joint training of a topic model.
    """

    def __init__(self, min_num_clients, logger=None):
        self.federation = federation.Federation()
        self.min_num_clients = min_num_clients
        self.minibatch = -1
        self.current_epoch = -2
        self.id_server = "IDS" + "_" + str(round(time.time()))
        self.path_global_corpus = get_tmpfile(str(self.id_server))

        # Create logger object
        if logger:
            self.logger = logger
        else:
            import logging
            FMT = '[%(asctime)-15s] [%(filename)s] [%(levelname)s] %(message)s'
            logging.basicConfig(format=FMT, level='INFO')
            self.logger = logging.getLogger('Server')

    def record_client_consensus(self, context, path_tmp_local_corpus):
        """
        Method to record the communication between a server and one of the clients in the federation at the time the clients first try to send its local vocabulary.

        Parameters
        ----------
        context : AuthMetadataContext
            An AuthMetadataContext providing information on the RPC
        path_tmp_local_corpus: pathlib.Path
            Path to the temporary file where the local corpus of the client currently under communication is located
        """

        def unregister_client():
            """No-parameter callable to be called on RPC termination, that invokes the Federation's disconnect method when a client finishes a communication with the server.
            """
            print(self.federation.federation_clients)
            client = \
                federation_client.FederationClient.get_pos_by_key(
                    context.peer(), self.federation.federation_clients)
            self.federation.federation_clients[client].vocab_sent = True
            self.federation.disconnect(context.peer())

        context.add_callback(unregister_client)
        self.federation.connect_consensus(
            context.peer(), path_tmp_local_corpus)

    def record_client(self, context, gradient, current_mb, current_iter, current_id_msg, num_max_iter):
        """
        Method to record the communication between a server and one of the clients in the federation at the time the clients send its updates.

        Parameters
        ----------
        context : AuthMetadataContext
            An AuthMetadataContext providing information on the RPC
        gradient: Pytorch.Tensor
            Gradient that the client is sending to the server at "current_iter" on "current_id_msg"
        current_mb: int
            Minibatch that corresponds with the gradient that is being sent by the client
        current_iter: int
            Epoch that corresponds with the gradient that is being sent by the client
        current_id_msg: int
            Id of the message with which the gradient is being sent
        num_max_iter: int
            Number of epochs with which the model is being trained
        """

        def unregister_client():
            """No-parameter callable to be called on RPC termination, that invokes the Federation's disconnect method when a client finishes a communication with the server.
            """
            self.federation.disconnect(context.peer())

        context.add_callback(unregister_client)
        self.federation.connect_update(
            context.peer(), gradient, current_mb, current_iter, current_id_msg, num_max_iter)

    def record_client_waiting_or_consensus(self, context, waiting):
        """
        Method to record the communication between a server and one of the clients in the federation at the time the client is waiting for server to send the consensed vocabulary or during the waiting time of the consensed vocabulary sending

        Parameters
        ----------
        context : AuthMetadataContext
            An AuthMetadataContext providing information on the RPC
        waiting: bool
            Whether the client is waiting for the vocabulary consensus sending or the vocabulary consensus is already happening
        """

        def unregister_client():
            """No-parameter callable to be called on RPC termination, that invokes the Federation's disconnect method when a client finishes a communication with the server.
            """
            self.federation.disconnect(context.peer())

        context.add_callback(unregister_client)
        self.federation.connect_waiting_or_consensus(context.peer(), waiting)

    def upload(self, request_iterator, context):

        path_tmp_local_corpus = get_tmpfile(context.peer())
        self.record_client_consensus(context, path_tmp_local_corpus)

        # Save chunks to temporal file
        save_chunks_to_file(request_iterator, path_tmp_local_corpus)

        return federated_pb2.Reply(length=os.path.getsize(path_tmp_local_corpus))

    def download(self, request, context):
        # Record clients waiting for the consensed request
        self.record_client_waiting_or_consensus(context, False)

        # Wait until all the clients in the federation have sent its local corpus
        wait(lambda: self.can_send_aggragated_vocab(), timeout_seconds=120,
             waiting_for="Aggregated vocab can be sent")

        # Combine clients vocabulary
        merged_corpus = []
        for client in self.federation.federation_clients:
            tmp_file_name = client.path_tmp_local_corpus
            with open(tmp_file_name, 'r') as f:
                for l in f.readlines():
                    merged_corpus.append(l.split())

        # Save in file and send it to clients
        save_corpus_in_file(merged_corpus, self.path_global_corpus)

        return get_file_chunks(self.path_global_corpus)

    def can_send_aggragated_vocab(self):
        """
        Checks whether all the clients meet the necessary condition for the sending of the consensed vocabulary.
        """
        if len(self.federation.federation_clients) < self.min_num_clients:
            return False
        for client in self.federation.federation_clients:
            if not client.vocab_sent:
                return False
        return True

    def sendLocalTensor(self, request, context):
        """
        Sends an ACK response to the client after receiving his local tensor update.

        Parameters
        ----------
        request: ClientTensorRequest
            Request of the client for sending an iteration's gradient
        context: AuthMetadataContext
            Context of the RPC communication between the client and the server

        Returns
        -------
        response : ServerReceivedResponse
            Server's response to confirm a client that its sent gradient was received
        """

        header = \
            federated_pb2.MessageHeader(id_response=self.id_server,
                                        id_to_request=request.header.id_request,
                                        message_type=federated_pb2.MessageType.SERVER_CONFIRM_RECEIVED)

        # Deserialize gradient update from the client and save in the corresponding Client object as numpy array
        dims = tuple([dim.size for dim in request.data.tensor_shape.dim])
        dtype_send = get_type_from_string(request.data.dtype)
        deserialized_bytes = np.frombuffer(
            request.data.tensor_content, dtype=dtype_send)
        deserialized_numpy = np.reshape(deserialized_bytes, newshape=dims)

        # Record client in the federation
        self.record_client(context,
                           deserialized_numpy,
                           request.metadata.current_mb,
                           request.metadata.current_epoch,
                           request.header.id_request,
                           request.metadata.num_max_epochs)

        id_client = \
            federation_client.FederationClient.get_pos_by_key(
                context.peer(),
                self.federation.federation_clients)

        # Update minibatch and epoch in the server
        self.current_minibatch = request.metadata.current_mb
        self.current_epoch = request.metadata.current_epoch

        # TODO
        federation_client.FederationClient.set_can_get_update_by_key(
            context.peer(),
            self.federation.federation_clients,
            False)

        response = federated_pb2.ServerReceivedResponse(header=header)
        return response

    def can_send_update(self):
        """Checks the conditions that need to be fullfilled in order to send the average tensor update to the clients.

        Returns:
        --------
            * boolean: True if the aggragate update can be sent.
        """

        if len(self.federation.federation_clients) < self.min_num_clients:
            return False
        for client in self.federation.federation_clients:
            if not client.can_get_update:
                return False
        return True

    def sendAggregatedTensor(self, request, context):
        """
        Sends the average update to the correspinding client based on the context of the gRPC channel.

        Parameters
        ----------
        request: federated_pb2.Empty
            Empty protocol buffer coming from the client
        context: AuthMetadataContext
            Context of the RPC communication between the client and the server

        Returns
        -------
        request: federated_pb2.ServerAggregatedTensorReques
            Request with the aggregated tensor
        """

        # Record clients waiting
        self.record_client_waiting_or_consensus(context, True)

        federation_client.FederationClient.set_can_get_update_by_key(
            context.peer(),
            self.federation.federation_clients,
            True)

        # Wait until all the clients in the federation have sent its current iteration gradient
        wait(lambda: self.can_send_update(), timeout_seconds=120,
             waiting_for="Update can be sent")

        # print(self.federation.federation_clients[0].tensor.shape)
        # print(self.federation.federation_clients[1].tensor.shape)
        # print(self.federation.federation_clients[2].tensor.shape)
        # print(self.federation.federation_clients[3].tensor.shape)
        # print(self.federation.federation_clients[4].tensor.shape)

        # Calculate average
        clients_tensors = [
            client.tensor for client in self.federation.federation_clients]
        average_tensor = np.mean(np.stack(clients_tensors), axis=0)
        print("The average tensor is: ", average_tensor)

        # Serialize tensor
        content_bytes = average_tensor.tobytes()
        size = federated_pb2.TensorShape()
        num_dims = len(average_tensor.shape)
        for i in np.arange(num_dims):
            name = "dim" + str(i)
            size.dim.extend(
                [federated_pb2.TensorShape.Dim(size=average_tensor.shape[i], name=name)])

        # Get the client to sent the update to based on the context of the gRPC channel
        client_to_repond = \
            federation_client.FederationClient.get_pos_by_key(
                context.peer(),
                self.federation.federation_clients)

        # Create request
        update_name = "Update for client with key " + \
            str(self.federation.federation_clients[client_to_repond].federation_key) + \
            " for the iteration " + \
            str(
                self.federation.federation_clients[client_to_repond].current_epoch)

        data = federated_pb2.Update(tensor_name=update_name,
                                    tensor_shape=size, tensor_content=content_bytes)
        # Send update
        header = federated_pb2.MessageHeader(id_response=self.id_server,
                                             message_type=federated_pb2.MessageType.SERVER_AGGREGATED_TENSOR_SEND)
        print(update_name)

        id_client = federation_client.FederationClient.get_pos_by_key(
            context.peer(),
            self.federation.federation_clients)

        request = federated_pb2.ServerAggregatedTensorRequest(
            header=header, data=data)
        return request
