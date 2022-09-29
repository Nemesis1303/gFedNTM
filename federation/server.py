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
from models.federated.federated_avitm import FederatedAVITM
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import FeatureUnion
from utils.auxiliary_functions import (deserializeNumpy,
                                       modelStateDict_to_proto,
                                       optStateDict_to_proto)
from waiting import wait

from federation import (federated_pb2, federated_pb2_grpc, federation,
                        federation_client)

GRPC_TRACE = all


class FederatedServer(federated_pb2_grpc.FederationServicer):
    """Class that describes the behaviour of the GRPC server to which several clients are connected to create a federation for the joint training of a topic model.
    """

    def __init__(self, min_num_clients, model_params, model_type, logger=None):
        self.federation = federation.Federation()
        self.min_num_clients = min_num_clients
        self.current_minibatch = -1
        self.current_epoch = -2
        self.id_server = "IDS" + "_" + str(round(time.time()))
        self.path_global_corpus = get_tmpfile(str(self.id_server))
        self.dicts = []

        self.global_model = None
        self.model_type = model_type
        self.model_parameters = model_params
        self.global_vocab = None

        # Create logger object
        if logger:
            self.logger = logger
        else:
            import logging
            FMT = '[%(asctime)-15s] [%(filename)s] [%(levelname)s] %(message)s'
            logging.basicConfig(format=FMT, level='INFO')
            self.logger = logging.getLogger('Server')

    def record_client_consensus(self, context, path_tmp_local_corpus, nr_samples):
        """
        Method to record the communication between a server and one of the clients in the federation at the time the clients first try to send its local vocabulary.

        Parameters
        ----------
        context : AuthMetadataContext
            An AuthMetadataContext providing information on the RPC
        path_tmp_local_corpus: pathlib.Path
            Path to the temporary file where the local corpus of the client currently under communication is located
        nr_samples: Number of client's data points 
        """

        def unregister_client():
            """No-parameter callable to be called on RPC termination, that invokes the Federation's disconnect method when a client finishes a communication with the server.
            """
            print(self.federation.federation_clients)
            client = \
                federation_client.FederationClient.get_pos_by_key(
                    context.peer(), self.federation.federation_clients)
            self.federation.federation_clients[client].vocab_sent = True
            self.federation.federation_clients[client].nr_samples = nr_samples
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

    def sendLocalDic(self, request, context):
        """
        Sends an ACK response to the client after receiving his local dictionary update.

        Parameters
        ----------
        request: ClientTensorRequest
            Request of the client for sending its vocabulary dictionary
        context: AuthMetadataContext
            Context of the RPC communication between the client and the server

        Returns
        -------
        response : ServerReceivedResponse
            Server's response to confirm a client that its sent vocabulary dictionary was received
        """

        path_tmp_local_corpus = get_tmpfile(context.peer())

        # Get client's vocab and save it dicts list
        vocab = dict([(pair.key, pair.value.ivalue)
                     for pair in request.pairs])
        self.dicts.append(vocab)

        cv = CountVectorizer(vocabulary=vocab)
        nr_samples = len(cv.get_feature_names_out())

        self.record_client_consensus(context, path_tmp_local_corpus, nr_samples)

        return federated_pb2.Reply(length=len(vocab))

    def sendGlobalDicAndInitialNN(self, request, context):
        """
        Sends the common vocabulary and the initialized NN to the correspinding client based on the context of the gRPC channel.

        Parameters
        ----------
        request: federated_pb2.Empty
            Empty protocol buffer coming from the client
        context: AuthMetadataContext
            Context of the RPC communication between the client and the server

        Returns
        -------
        request: federated_pb2.ServerAggregatedTensorReques
            Request with the common vocabulary and the initialized NN
        """

        # Record clients waiting for the consensed request
        self.record_client_waiting_or_consensus(context, False)

        # Wait until all the clients in the federation have sent its local vocab
        wait(lambda: self.can_send_aggragated_vocab(), timeout_seconds=120,
             waiting_for="Aggregated vocab can be sent")

        # Get init_size
        vocabs = []
        for dic_i in self.dicts:
            cv_i = CountVectorizer(vocabulary=dic_i)
            name_i = "CV" + str(dic_i)
            vocabs.append((name_i, cv_i))
        self.global_vocab = FeatureUnion(vocabs)
        idx2token = self.global_vocab.get_feature_names_out()
        self.model_parameters["input_size"] = len(idx2token)
        self.model_parameters["id2token"] = \
            {k: v for k, v in zip(range(0, len(idx2token)), idx2token)}

        # Create initial NN
        self.logger.info("Server initializing global model")
        if self.model_type == "prod":
            self.global_model = \
                FederatedAVITM(self.model_parameters, self, self.logger)
        elif self.model_type == "ctm":
            print("To be implemented")
        else:
            self.logger.error("Provided underlying model not supported")

        modelUpdate_ = \
            modelStateDict_to_proto(self.global_model.model.state_dict(), -1)
        optUpdate_ = \
            optStateDict_to_proto(self.global_model.optimizer.state_dict())
        nNUpdate = federated_pb2.NNUpdate(
            modelUpdate=modelUpdate_,
            optUpdate=optUpdate_
        )

        feature_union = federated_pb2.FeatureUnion(
            initialNN=nNUpdate)
        # Serialize clients vocab
        for vocab_dict in self.dicts:
            dic = federated_pb2.Dictionary()
            for key_, value_ in vocab_dict.items():
                dic.pairs.extend(
                    [federated_pb2.Dictionary.Pair(key=key_, value=federated_pb2.Dictionary.Pair.Value(ivalue=value_))])
            feature_union.dic.extend([dic])

        return feature_union

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

        # Deserialize gradient updates from the client and save then in the corresponding Client object as numpy array
        gradients = {}
        for update in request.updates:
            deserialized_numpy = deserializeNumpy(update.tensor)
            gradients[update.tensor_name] = deserialized_numpy

        # Record client in the federation
        self.record_client(context,
                           gradients,
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
            if client.current_mb != self.current_minibatch:
                print("DIFERENT")
                print(client.current_mb)
                print(self.current_minibatch)
                #return False
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
        request: federated_pb2.ServerAggregatedTensorRequest
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

        # Calculate average for each tensor
        # Get dict of tensor, of entry per update
        keys = self.federation.federation_clients[0].tensors.keys()
        averages = {}
        for key in keys:
            clients_tensors = [
                client.tensors[key]*client.nr_samples for client in self.federation.federation_clients]
            average_tensor = np.mean(np.stack(clients_tensors), axis=0)
            averages[key] = average_tensor
            print("The average tensor " + key + " is: ", average_tensor)

        # Peform updates
        # TODO: Update for different models
        self.global_model.optimize_on_minibatch_from_server(averages)

        modelUpdate_ = \
            modelStateDict_to_proto(self.global_model.model.state_dict(), -1)
        optUpdate_ = \
            optStateDict_to_proto(self.global_model.optimizer.state_dict())
        nNUpdate = federated_pb2.NNUpdate(
            modelUpdate=modelUpdate_,
            optUpdate=optUpdate_
        )

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

        # Send update
        header = federated_pb2.MessageHeader(id_response=self.id_server,
                                             message_type=federated_pb2.MessageType.SERVER_AGGREGATED_TENSOR_SEND)
        print(update_name)

        id_client = federation_client.FederationClient.get_pos_by_key(
            context.peer(),
            self.federation.federation_clients)

        request = federated_pb2.ServerAggregatedTensorRequest(
            header=header, nndata=nNUpdate)
        
        return request
