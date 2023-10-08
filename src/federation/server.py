# -*- coding: utf-8 -*-
"""
This script contains the class that defines the server-side part of the federated learning process, assuming a centralized federated scenario. To synchronize the clients' updates sending, the training has been divided into two phases:

* PHASE 1: VOCABULARY CONSENSUS PHASE. The server acts as a coordinator and the clients send their local vocabularies to the server. The server computes the consensus vocabulary and sends it to the clients. The clients wait until they receive the consensus vocabulary to start the training.
* PHASE 2: FEDERATED TRAINING PHASE. For this stage, the server does not behave as a server per definition. Instead, it acts as a client to request the gradient updates from the 'Client-Server' created on each client's side.

Created on Feb 1, 2022
Last updated on Aug 24, 2023

.. codeauthor:: L. Calvo-Bartolomé (lcalvo@pa.uc3m.es)
"""

from collections import OrderedDict
import sys
import threading
import time


import grpc
import torch
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from src.federation import federation, federation_client
from src.models.federated.federated_avitm import FederatedAVITM
from src.models.federated.federated_ctm import FederatedCTM
from src.protos import federated_pb2, federated_pb2_grpc
from src.utils.auxiliary_functions import (deserializeNumpy,
                                           modelStateDict_to_proto,
                                           optStateDict_to_proto)
import logging
from waiting import wait

GRPC_TRACE = all


class FederatedServer(federated_pb2_grpc.FederationServicer):
    """Class that describes the behaviour of the GRPC server to which several clients are connected to create a federation for the joint training of a topic model.

    Parameters
    ----------
    min_num_clients : int
        Minimum number of clients to start the federation
    model_params: dict
        Dictionary with the parameters of the topic model
    model_type: str
        Underlying topic modeling algorithm with which the federated topic model is going to be constructed (prod|ctm)
    save_server: str
        Path to to save the global model of the federated topic model
    client_server_addres: str
        Base address of a client server
    base_port: int
        Base port of a client-server, which which will be generated as follows: str(base_port + id)
    logger: logging.Logger
        Logger object to log messages
    """

    # Lock to synchronize the start of the federated training
    _start_federated_training_lock = threading.Lock()
    _is_federated_training_started = False

    def __init__(
        self,
        min_num_clients: int,
        model_params: dict,
        model_type: str,
        max_iters: int,
        opts_client: dict,
        save_server: str,
        logs_server: str,
        grads_to_share: list[str] = ["prior_mean", "prior_variance", "beta"],
        client_server_addres: str = "gfedntm-client",
        base_port: int = 50051,
        logger: logging.Logger = None
    ) -> None:

        # Define attributes
        self._min_num_clients = min_num_clients
        self._model_parameters = model_params
        self._model_type = model_type
        self._max_iters = max_iters
        self._opts_client = opts_client
        self._grads_to_share = grads_to_share
        self._client_server_addres = client_server_addres
        self._save_server = save_server
        self._logs_server = logs_server
        self._base_port = base_port

        # Create logger object
        if logger:
            self._logger = logger
        else:
            FMT = logging.Formatter(
                "%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")

            self._logger = logging.getLogger('Client')
            self._logger.setLevel(logging.DEBUG)

            fileHandler = logging.FileHandler(filename=logs_server)
            fileHandler.setFormatter(FMT)
            self._logger.addHandler(fileHandler)

            consoleHandler = logging.StreamHandler(sys.stdout)
            consoleHandler.setFormatter(FMT)
            self._logger.addHandler(consoleHandler)

        # Define additional attributes
        self._id_server = "IDS" + "_" + str(round(time.time()))
        self._dicts = []
        self._global_model = None
        self._global_vocab = None
        self._federation = federation.Federation()

    def record_client_consensus(
        self,
        context: grpc.AuthMetadataContext,
        nr_samples: int
    ) -> None:
        """
        Method to record the communication between a server and one of the clients in the federation at the time the clients first try to send its local vocabulary.

        Parameters
        ----------
        context : grpc.AuthMetadataContext
            An AuthMetadataContext providing information on the RPC
        nr_samples: int
            Number of client's documents
        """

        def unregister_client():
            """No-parameter callable to be called on RPC termination, that invokes the Federation's disconnect method when a client finishes a communication with the server.
            """
            self._logger.info(
                f"-- -- Unregistering client{context.peer()} from consensus")
            client = \
                federation_client.FederationClient.get_pos_by_key(
                    context.peer(), self._federation.federation_clients)
            self._federation.federation_clients[client].vocab_sent = True
            self._federation.federation_clients[client].nr_samples = nr_samples
            self._federation.disconnect(context.peer())

        context.add_callback(unregister_client)
        self._federation.connect_consensus(context.peer())

        return

    def record_client_waiting_or_consensus(
        self,
        context: grpc.AuthMetadataContext,
        waiting: bool
    ) -> None:
        """
        Method to record the communication between a server and one of the clients in the federation at the time the client is waiting for server to send the consensed vocabulary or during the waiting time of the consensed vocabulary sending

        Parameters
        ----------
        context : grpc.AuthMetadataContext
            An AuthMetadataContext providing information on the RPC
        waiting: bool
            Whether the client is waiting for the vocabulary consensus sending or the vocabulary consensus is already happening
        """

        def unregister_client():
            """No-parameter callable to be called on RPC termination, that invokes the Federation's disconnect method when a client finishes a communication with the server.
            """
            self._logger.info(
                f"-- -- Unregistering client{context.peer()} from waiting or consensus")
            self._federation.disconnect(context.peer())

        context.add_callback(unregister_client)
        self._federation.connect_waiting_or_consensus(context.peer(), waiting)

        return

    def sendLocalDic(
        self,
        request: federated_pb2.ClientTensorRequest,
        context: grpc.AuthMetadataContext
    ) -> federated_pb2.ServerReceivedResponse:
        """
        Sends an ACK response to the client after receiving his local dictionary update.

        Parameters
        ----------
        request: federated_pb2.ClientTensorRequest
            Request of the client for sending its vocabulary dictionary
        context: grpc.AuthMetadataContext
            Context of the RPC communication between the client and the server

        Returns
        -------
        response : ServerReceivedResponse
            Server's response to confirm a client that its sent vocabulary dictionary was received
        """

        # Get client's vocab and save it dicts list
        vocab = dict([(pair.key, pair.value.ivalue)
                     for pair in request.vocab.pairs])
        self._dicts.append(vocab)

        # Record clients waiting for the consensed request
        self.record_client_consensus(context, request.nr_samples)

        # Update client's information in the federation
        federation_client.FederationClient.set_id_by_key(
            key=context.peer(),
            federation_clients=self._federation.federation_clients,
            id=request.client_id)

        return federated_pb2.Reply(length=len(vocab))

    def sendGlobalDicAndInitialNN(
        self,
        request: federated_pb2.Empty,
        context: grpc.AuthMetadataContext
    ) -> federated_pb2.FeatureUnion:
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
        request: federated_pb2.FeatureUnion
            Request with the common vocabulary and the initialized NN
        """

        # Record clients waiting for the consensed request
        self.record_client_waiting_or_consensus(context, False)

        # Wait until all the clients in the federation have sent its local vocab
        wait(lambda: self.can_send_aggragated_vocab(), timeout_seconds=120,
             waiting_for="Aggregated vocab can be sent")

        # Serialize model_params
        model_params_dic = federated_pb2.Dictionary()

        for key_, value_ in self._model_parameters.items():
            if isinstance(value_, str):
                el = federated_pb2.Dictionary.Pair(
                    key=key_, value=federated_pb2.Dictionary.Pair.Value(svalue=value_))
            elif value_ is None:
                el = federated_pb2.Dictionary.Pair(
                    key=key_, value=federated_pb2.Dictionary.Pair.Value(svalue=str(value_)))
            elif isinstance(value_, float):
                el = federated_pb2.Dictionary.Pair(
                    key=key_, value=federated_pb2.Dictionary.Pair.Value(fvalue=value_))
            elif isinstance(value_, bool):
                el = federated_pb2.Dictionary.Pair(
                    key=key_, value=federated_pb2.Dictionary.Pair.Value(bvalue=value_))
            elif isinstance(value_, tuple):
                tuple_ = federated_pb2.Tuple()
                for el in list(value_):
                    tuple_.values.extend(
                        [federated_pb2.Tuple.Valuet(ivalue=el)])
                el = federated_pb2.Dictionary.Pair(
                    key=key_, value=federated_pb2.Dictionary.Pair.Value(tvalue=tuple_))
            else:
                if isinstance(value_, int):
                    el = federated_pb2.Dictionary.Pair(
                        key=key_, value=federated_pb2.Dictionary.Pair.Value(ivalue=value_))
            model_params_dic.pairs.extend([el])

        # Get init_size
        vocabs = []
        for i in range(len(self._dicts)):
            vocab_i = [key for key in self._dicts[i].keys()]
            vocabs = list(set(vocabs) | set(vocab_i))
        vocabs = sorted(vocabs)

        # Create vocabulary dictionary
        vocabulary_dict = {}
        for i, t in enumerate(vocabs):
            vocabulary_dict[t] = i

        # Create CountVectorizer from vocabulary dictionary
        cv_g = CountVectorizer(vocabulary=vocabulary_dict)

        # Save information of input_size and id2token in model_params
        idx2token = cv_g.get_feature_names_out()
        self._model_parameters["input_size"] = len(idx2token)
        self._model_parameters["id2token"] = \
            {k: v for k, v in zip(range(0, len(idx2token)), idx2token)}

        # Create initial global federated topic models
        self._logger.info("-- -- Server initializing global model")
        if self._model_type == "avitm":
            self._global_model = \
                FederatedAVITM(self._model_parameters, self._grads_to_share. self._logger)
            self._logger.info("-- -- AVITM model initialized")
        elif self._model_type == "ctm":
            self._global_model = \
                FederatedCTM(self._model_parameters, self._grads_to_share, self._logger)
            self._logger.info("-- -- CTM model initialized")
        else:
            self._logger.error("Provided underlying model not supported")

        modelUpdate_ = \
            modelStateDict_to_proto(
                self._global_model.model.state_dict(), -1, self._model_type)
        optUpdate_ = \
            optStateDict_to_proto(self._global_model.optimizer.state_dict())
        nNUpdate = federated_pb2.NNUpdate(
            modelUpdate=modelUpdate_,
            optUpdate=optUpdate_
        )

        self._logger.info(
            "-- -- modelStateDict_to_proto and optStateDict created ")

        feature_union = federated_pb2.FeatureUnion(
            initialNN=nNUpdate,
            model_params=model_params_dic,
            model_type=self._model_type)

        # Serialize clients vocab
        dic = federated_pb2.Dictionary()
        for key_, value_ in vocabulary_dict.items():
            dic.pairs.extend(
                [federated_pb2.Dictionary.Pair(key=key_, value=federated_pb2.Dictionary.Pair.Value(ivalue=value_))])
        feature_union.dic.extend([dic])

        self._logger.info(
            "-- -- Client vocab serialized")

        return feature_union

    def can_send_aggragated_vocab(self) -> bool:
        """
        Checks whether all the clients meet the necessary condition for the sending of the consensed vocabulary.

        Returns
        -------
        boolean: bool
            True if the vocabulary can be sent
        """
        if len(self._federation.federation_clients) < self._min_num_clients:
            return False
        for client in self._federation.federation_clients:
            if not client.vocab_sent:
                return False
        return True

    def can_start_training(self) -> bool:
        """Checks the conditions that need to be fullfilled in order to start training phase

        Returns
        -------
        boolean : bool
            True if the federated training can start
        """

        if len(self._federation.federation_clients) < self._min_num_clients:
            return False
        for client in self._federation.federation_clients:
            if not client.ready_for_training:
                return False
        return True

    def trainFederatedModel(
        self,
        request: federated_pb2.Empty,
        context: grpc.AuthMetadataContext
    ) -> federated_pb2.Empty:
        """
        Trains the federated model by requestnig the gradients from them.

        Parameters
        ----------
        request: federated_pb2.Empty
            Empty protocol buffer coming from the client
        context: AuthMetadataContext
            Context of the RPC communication between the client and the server

        Returns
        -------
        request: federated_pb2.Empty()
            Empty request
        """

        # Record clients waiting for the training phase to start
        self.record_client_waiting_or_consensus(context, True)

        # Set current client to be ready for training
        federation_client.FederationClient.set_can_start_training(
            context.peer(),
            self._federation.federation_clients,
            True)

        # Wait for all clients to be ready for training
        wait(lambda: self.can_start_training(), timeout_seconds=120,
             waiting_for="Training to start")

        # Start training phase
        self._logger.info(f"-- -- Starting training phase")
        with FederatedServer._start_federated_training_lock:
            if not FederatedServer._is_federated_training_started:
                threading.Thread(target=self.do_federated_training).start()
                FederatedServer._is_federated_training_started = True

        return federated_pb2.Empty()

    def do_federated_training(self) -> None:
        """Trains the federated model by requesting the gradients from the clients.
        """

        def get_address_to_connect(client):
            return self._client_server_addres + \
                str(client.client_id) + ":" + \
                str(self._base_port + client.client_id)

        def sleep_until_next_client(time_sleep=3):
            self._logger.info(
                f"-- -- Server entering into sleep mode before next client request...")
            time.sleep(time_sleep)

        def sleep_until_stop(time_sleep=10):
            self._logger.info(
                f"-- -- Server {self._id_server} finished training. Waiting for saving")
            time.sleep(time_sleep)

        # START SERVER AS 'CLIENT'
        # Open channel for communication with the ClientServer
        # Iter over max iters to train federated model
        for i in range(self._max_iters):

            print("••••••••••••••••••••••••••")
            print("Global iteration: ", i)
            print("••••••••••••••••••••••••••")

            clients_s = [self._federation.federation_clients[client_pos]
                         for client_pos in range(len(self._federation.federation_clients))]

            ###############################################################
            # Request gradients
            ###############################################################
            for client in clients_s:

                address_to_connect = get_address_to_connect(client)

                self._logger.info(
                    f"-- -- Requesting gradient from client id {str(client.client_id)} at address {address_to_connect}")

                with grpc.insecure_channel(address_to_connect, options=self._opts_client) as channel:
                    stub = federated_pb2_grpc.FederationServerStub(channel)

                    # Request the gradient from the client i
                    grad_req = federated_pb2.ServerGetGradientRequest(iter=i)

                    if stub:
                        # Get the gradient from the client i
                        client_tensor_req = stub.getGradient(grad_req)

                        # Deserialize clients' gradient updates
                        gradients = {update.tensor_name: deserializeNumpy(
                            update.tensor) for update in client_tensor_req.updates}

                        # Save the gradients in the corresponding Client object
                        client.update_client_state(
                            gradients,
                            client_tensor_req.metadata.current_mb,
                            client_tensor_req.metadata.current_epoch, client_tensor_req.header.id_request)
                        self._logger.info(
                            f"-- -- Connecting to {address_to_connect} worked as expected")

                # Sleep before next client request
                sleep_until_next_client()

            ###############################################################
            # Average gradients
            ###############################################################
            keys = [key for key in client.tensors.keys(
            ) if key in self._grads_to_share]
            averages = {}
            for key in keys:
                N = np.array(
                    [client.nr_samples for client in self._federation.federation_clients])
                clients_tensors = [
                    client.tensors[key]*client.nr_samples for client in self._federation.federation_clients]
                average_tensor = np.sum(
                    np.stack(clients_tensors), axis=0) / N.sum()
                averages[key] = average_tensor

            # Create model update
            state_dict = OrderedDict({k: torch.tensor(v)
                                     for k, v in averages.items()})
            modelUpdate_ = modelStateDict_to_proto(
                state_dict, -1, self._model_type)

            nNUpdate = federated_pb2.NNUpdate(
                modelUpdate=modelUpdate_,
            )

            # Send Aggregated request to Server-Clients
            for client in clients_s:

                address_to_connect = get_address_to_connect(client)

                self._logger.info(
                    f"-- -- Sending aggragated request to client id {str(client.client_id)} at address {address_to_connect}")

                # Create request
                header = federated_pb2.MessageHeader(
                    id_response=str(self._id_server),
                    message_type=federated_pb2.MessageType.SERVER_AGGREGATED_TENSOR_SEND)

                agg_request = federated_pb2.ServerAggregatedTensorRequest(
                    header=header, nndata=nNUpdate)

                with grpc.insecure_channel(address_to_connect, options=self._opts_client) as channel:
                    stub = federated_pb2_grpc.FederationServerStub(channel)

                    if stub:
                        # get ack confirming received ok
                        ack = stub.sendAggregatedTensor(agg_request)
                self._logger.info(f"-- -- ACK received for iter {i}")

        # Save global model
        self._global_model.get_topics_in_server(self._save_server)

        # Sleep before STOP request
        sleep_until_stop()

        self._logger.info(
            f"-- -- Server {self._id_server} sending stop training request to client servers...")

        for client in clients_s:
            address_to_connect = get_address_to_connect(client)

            self._logger.info(
                f"-- -- Sending stop request to client id {str(client.client_id)} at address {address_to_connect}")

            # Create request
            header = federated_pb2.MessageHeader(
                id_response=str(self._id_server),
                message_type=federated_pb2.MessageType.SERVER_STOP_TRAINING_REQUEST)
            stop_request = federated_pb2.ServerAggregatedTensorRequest(
                header=header)

            with grpc.insecure_channel(address_to_connect, options=self._opts_client) as channel:
                stub = federated_pb2_grpc.FederationServerStub(channel)

                if stub:
                    # get ack confirming received ok
                    ack = stub.sendAggregatedTensor(stop_request)
                self._logger.info(f"-- -- ENDING ACK received")

        return
