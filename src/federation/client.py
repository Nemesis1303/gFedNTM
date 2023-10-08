# -*- coding: utf-8 -*-
"""
This script contains the two classes that define the client-side part of the federated learning process. 

* The CLIENT class: This class contains stubs that facilitate interactions with the server-side via a gRPC channel during the vocabulary consensus phase.

* The FEDERATEDCLIENTSERVER class: This class defines the behavior of the client when it acts as a 'client-server' to process server petitions for another gradient update. T

Created on Feb 1, 2022
Last updated on Aug 24, 2023

.. codeauthor:: L. Calvo-Bartolomé (lcalvo@pa.uc3m.es)
"""
import sys
import time
from typing import Union
import threading

import pandas as pd
import grpc
from concurrent import futures
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import warnings
from src.models.base.contextualized_topic_models.datasets.dataset import \
    CTMDataset
from src.models.base.contextualized_topic_models.utils.data_preparation import \
    TopicModelDataPreparation
from src.models.base.pytorchavitm.datasets.bow_dataset import BOWDataset
from src.models.federated.federated_avitm import FederatedAVITM
from src.models.federated.federated_ctm import FederatedCTM
from src.protos import federated_pb2, federated_pb2_grpc
from src.utils.auxiliary_functions import (proto_to_modelStateDict,
                                           proto_to_optStateDict,
                                           serializeTensor)

warnings.filterwarnings('ignore')
GRPC_TRACE = all

# ======================================================
# CLIENT-SERVER
# ======================================================
class FederatedClientServer(federated_pb2_grpc.FederationServerServicer):
    """Class that describes the behavior to be followed by the client when it acts as a 'client-server' to process the server petitions.
    """

    def __init__(
        self,
        local_model: Union[FederatedAVITM, FederatedCTM],
        train_data: Union[BOWDataset, CTMDataset],
        id: int,
        save_client: str,
        logger=None
    ):

        # Attributes
        self._local_model = local_model
        self.id = id
        self._save_client = save_client

        # Initialize all parameters needed for the training of the local model
        self._local_model.preFit(train_data)

        # Create logger object
        if logger:
            self._logger = logger
        else:
            import logging
            FMT = '[%(asctime)-15s] [%(filename)s] [%(levelname)s] %(message)s'
            logging.basicConfig(format=FMT, level='DEBUG',
                                filename="logs_client-server.txt")
            self._logger = logging.getLogger('ClientServer')
            self._logger.setLevel(logging.DEBUG)

        self._logger.info("FederatedClientServer object initialized")

    def getGradient(
        self,
        request: federated_pb2.ServerGetGradientRequest,
        context: grpc.AuthMetadataContext
    ) -> federated_pb2.ClientTensorRequest:
        """Receives a request to send a minibatch's gradient from the server and generates the response.

        Parameters
        ----------
        request: federated_pb2.ServerGetGradientRequest
            Request to send the minibatch's gradient
        context: AuthMetadataContext
            Context of the RPC communication between the ClientServer and the Server

        Returns
        -------
        response: federated_pb2.ClientTensorRequest
            Response with the minibatch's gradient
        """

        # Get global epoch from the request
        self.global_epoch = request.iter

        # Generate gradient update (train one minibatch of local model)
        gr_upt = self._local_model.train_mb_delta()

        # Get gradients
        gradients_ = [[key, gr_upt[key]] for key in gr_upt if key not in [
            "current_mb", "current_epoch", "num_epochs"]]

        # Generate response's header
        id_message = "ID" + str(self.id) + "_" + str(round(time.time()))
        header = federated_pb2.MessageHeader(
            id_request=id_message,
            message_type=federated_pb2.MessageType.CLIENT_TENSOR_SEND)

        # Generate response's metadata
        metadata = federated_pb2.MessageAdditionalData(
            current_mb=gr_upt["current_mb"],
            current_epoch=gr_upt["current_epoch"],
            num_max_epochs=gr_upt["num_epochs"],
            id_machine=int(self.id))

        # Generate Protos Update with the gradients
        updates_ = [federated_pb2.Update(
            tensor_name=gradient[0],
            tensor=serializeTensor(gradient[1])
        ) for gradient in gradients_]

        # Generate Protos ClientTensorRequest with the update
        response = federated_pb2.ClientTensorRequest(
            header=header, metadata=metadata, updates=updates_)

        self._logger.info(
            '-- -- Client %s sent gradient update for global epoch %s', str(self.id), self.global_epoch)

        return response

    def sendAggregatedTensor(
        self,
        request: federated_pb2.ServerAggregatedTensorRequest,
        context: grpc.AuthMetadataContext
    ) -> federated_pb2.ClientReceivedResponse:
        """Process the receival of the aggregated tensor from the server and generates an ACK message.

        Parameters
        ----------
        request: federated_pb2.ServerAggregatedTensorRequest
            Request with the aggregated tensor
        context: AuthMetadataContext
            Context of the RPC communication between the ClientServer and the Server

        Returns
        -------
        response: federated_pb2.ClientReceivedResponse
            ACK to the server
        """

        if request.header.message_type == federated_pb2.MessageType.SERVER_AGGREGATED_TENSOR_SEND:
            self._logger.info(
                '-- -- Client-server %s received aggregated tensor for global epoch %s',
                str(self.id), self.global_epoch)
            # Process request and update local model with server's weights
            modelStateDict = proto_to_modelStateDict(
                request.nndata.modelUpdate)

            # Update local model with server's weights
            self._local_model.deltaUpdateFit(
                modelStateDict=modelStateDict,
                save_dir=self._save_client)

            header = federated_pb2.MessageHeader(
                id_request=str(self.global_epoch),
                message_type=federated_pb2.MessageType.CLIENT_CONFIRM_RECEIVED)
            response = federated_pb2.ClientReceivedResponse(header=header)

        elif request.header.message_type == federated_pb2.MessageType.SERVER_STOP_TRAINING_REQUEST:
            self._logger.info(
                '-- -- Client-server %s received reququest for finished training',
                str(self.id))

            header = federated_pb2.MessageHeader(
                message_type=federated_pb2.MessageType.CLIENT_CONFIRM_RECEIVED)
            response = federated_pb2.ClientReceivedResponse(header=header)

            self._logger.info(f"-- -- Getting results at client side...")
            self._local_model.get_results_model(self._save_client)

        return response

# ======================================================
# CLIENT
# ======================================================
class Client:
    """ Class that describes the client's behaviour during the vocabulary consensus phase.
    """

    def __init__(
        self,
        id: int,
        stub: federated_pb2_grpc.FederationStub,
        local_corpus: list,
        data_type: str,
        opts_server: dict,
        save_client: str,
        logs_client: str,
        sleep_time: int,
        base_port: int,
        grads_to_share: list[str] = ["prior_mean", "prior_variance", "beta"],
        logger=None
    ):
        """
        Object's initializer

        Parameters
        ----------
        id : int
            Client's ide
        stub : federated_pb2_grpc.FederationStub
            Module acting as the interface for gRPC client
        local_corpus : List[str]
            List of documents that constitute the node's local corpus
        data_type : str
            Type of data used for the training of the local model
        opts_server : dict
            Dictionary with the options for the GRPC client-server
        save_client : str
            Path to save the client's model
        logs_client : str
            Path to save the client's logs
        base_port : int
            Base port to use for the client-server, which will be generated as follows: str(base_port + id)
        logger : logging.Logger, optional
            Logger object, by default None
        """

        self.id = id
        self._stub = stub
        self._save_client = save_client
        self._sleep_time = sleep_time
        self._base_port = base_port
        self._grads_to_share = grads_to_share

        # Create logger object
        if logger:
            self._logger = logger
        else:
            import logging
            FMT = logging.Formatter(
                "%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")

            self._logger = logging.getLogger('Client')
            self._logger.setLevel(logging.DEBUG)

            fileHandler = logging.FileHandler(filename=logs_client)
            fileHandler.setFormatter(FMT)
            self._logger.addHandler(fileHandler)

            consoleHandler = logging.StreamHandler(sys.stdout)
            consoleHandler.setFormatter(FMT)
            self._logger.addHandler(consoleHandler)

        # Get local corpus and embeddings (if CTM)
        self._local_corpus, self._local_embeddings = \
            self.__get_local_corpus(data_type, local_corpus)

        # Other attributes
        self._local_model_type = None
        self._model_parameters = None
        self.local_model = None
        self.train_data = None
        self._global_epoch = -3

        # Send vocabulary to server
        self.__send_local_vocab()

        # Wait for the consensed vocabulary and initial NN
        self.__wait_for_agreed_vocab_NN()

        # Create client_server in separate thread
        self.__start_client_server(opts_server, save_client)

        # Send ready for training after vocabulary consensus phase
        self.__send_ready_for_training()

    def __start_client_server(
        self,
        opts_server: dict,
        save_client: str
    ) -> None:
        """Starts the client-server in a separate thread.

        Parameters
        ----------
        opts_server : dict
            Dictionary with the options for the GRPC client-server
        save_client : str
            Path to save the client's model
        """

        # Create client-server
        client_server = grpc.server(
            futures.ThreadPoolExecutor(), options=opts_server)
        federated_server = FederatedClientServer(
            self.local_model, self.train_data, self.id, save_client, self._logger)
        federated_pb2_grpc.add_FederationServerServicer_to_server(
            federated_server, client_server)
        client_server.add_insecure_port(
            '[::]:' + str(self._base_port + self.id))
        client_server.start()

        # Start client-server thread
        federated_server._logger.info(
            f"-- -- Starting Client-server at {str(self._base_port + self.id)}...")
        try:
            thread = threading.Thread(
                target=client_server.wait_for_termination)
            thread.daemon = True
            thread.start()
            self._logger.info("-- -- Client-server thread started")
        except:
            self._logger.info("-- -- Could not start client-server thread")
        return

    def __get_local_corpus(
        self,
        data_type: str,
        corpus: pd.DataFrame,
    ) -> Union[list, np.ndarray]:
        """
        Gets the the local training corpus based on whether the input provided is synthetic or real.

        Parameters
        ----------
        data_type : str
            Type of data used for the training of the local model
        corpus : pd.DataFrame
            Dataframe with the local corpus

        Returns
        -------
        local_corpus : list
            List of documents that constitute the node's local corpus
        local_embeddings : np.ndarray
            Array with the embeddings of the local corpus
        """

        local_embeddings = None
        if data_type == "synthetic":
            local_corpus = \
                [" ".join(corpus[i]) for i in np.arange(len(corpus))]
        else:
            df_lemas = corpus[["bow_text"]].values.tolist()
            local_corpus = [' '.join(doc) for doc in df_lemas]
            if "embeddings" in list(corpus.columns.values):
                local_embeddings = corpus["embeddings"].values
                if isinstance(local_embeddings[0], str):
                    local_embeddings = np.array(
                        [np.array(el.split(), dtype=np.float32) for el in local_embeddings])
        return local_corpus, local_embeddings

    def __prepare_vocab_to_send(self) -> dict:
        """
        Gets the vocabulary associated to the local corpus as a dictionary object.

        Returns
        -------
        vocab_dict : dict
            Dictionary with the vocabulary associated to the local corpus
        """

        # Create a CountVectorizer object to convert a collection of text documents into a matrix of token counts
        cv = CountVectorizer(
            input='content', lowercase=True, stop_words='english', binary=False)

        # Learn the vocabulary dictionary, bow = document-term matrix
        cv.fit_transform(self._local_corpus)
        vocab_dict = cv.vocabulary_

        return vocab_dict

    def __send_local_vocab(self) -> None:
        """
        Sends the local vocabulary to the server and waits for its ACK.
        """

        # Get vocabulary to sent
        vocab_dict = self.__prepare_vocab_to_send()

        # Protofy vocabulary
        dic = federated_pb2.Dictionary()
        for key_, value_ in vocab_dict.items():
            dic.pairs.extend(
                [federated_pb2.Dictionary.Pair(key=key_, value=federated_pb2.Dictionary.Pair.Value(ivalue=value_))])

         # Construct request to send
        req = federated_pb2.DictRequest(vocab=dic,
                                        client_id=self.id,
                                        nr_samples=len(self._local_corpus))

        # Send dictionary to the server and wait for his response
        if self._stub:
            response = self._stub.sendLocalDic(req)
            self._logger.info(
                '-- -- Client %s vocab is being sent to server.', str(self.id))
            if response.length == len(vocab_dict):
                self._logger.info(
                    '-- -- Server received correctly vocab from client %s.', str(self.id))

        return

    def __wait_for_agreed_vocab_NN(self) -> None:
        """
        Waits while receiving the agreed vocabulary sent by the server.
        """

        self._logger.info(
            '-- -- Client %s receiving consensus vocab and initial NN.',
            str(self.id))

        # Get global model_parameters, model_type, dictionary and initialized NN
        response = self._stub.sendGlobalDicAndInitialNN(federated_pb2.Empty())

        self._logger.info(
            '-- -- Client %s response received.',
            str(self.id))

        # Unprotofy model params and model type
        model_params_aux = []
        for pair in response.model_params.pairs:
            if pair.value.HasField("svalue"):
                if pair.value.svalue == "None":
                    model_params_aux.append((pair.key, None))
                else:
                    model_params_aux.append((pair.key,  pair.value.svalue))
            elif pair.value.HasField("ivalue"):
                model_params_aux.append((pair.key, pair.value.ivalue))
            elif pair.value.HasField("fvalue"):
                model_params_aux.append((pair.key, pair.value.fvalue))
            elif pair.value.HasField("tvalue"):
                tuple_i = tuple([el.ivalue for el in pair.value.tvalue.values])
                model_params_aux.append((pair.key, tuple_i))
            elif pair.value.HasField("bvalue"):
                model_params_aux.append((pair.key, pair.value.bvalue))
        self._model_parameters = dict(model_params_aux)
        self._local_model_type = response.model_type

        self._logger.info(
            '-- -- Client %s received model params and model type.',
            str(self.id))

        # Unprotofy global dictionary
        vocabs = []
        for dic_i in range(len(response.dic)):
            vocab_i = dict([(pair.key, pair.value.ivalue)
                            for pair in response.dic[dic_i].pairs])
            vocabs.append(vocab_i)

        self._global_vocab = CountVectorizer(vocabulary=vocabs[0])
        self._logger.info(
            '-- -- Client %s model vocab unprotofied.',
            str(self.id))

        # Local document-term matrix as function of the global vocabulary
        train_bow = self._global_vocab.transform(
            self._local_corpus).toarray()

        # Array mapping from feature integer indices to feature name
        idx2token = self._global_vocab.get_feature_names_out()
        self._model_parameters["input_size"] = len(idx2token)
        id2token = {k: v for k, v in zip(range(0, len(idx2token)), idx2token)}
        self._model_parameters["id2token"] = id2token

        if self._local_model_type == "avitm":

            # The train dataset is an object from the class BOWDataset
            self.train_data = BOWDataset(train_bow, idx2token)

            # Initialize FederatedAVITM
            self.local_model = \
                FederatedAVITM(self._model_parameters, self._grads_to_share, self._logger)

        elif self._local_model_type == "ctm":

            # The train dataset is an object from the class CTMDataset
            qt = TopicModelDataPreparation()
            qt.vectorizer = self._global_vocab
            qt.id2token = id2token
            qt.vocab = idx2token
            self.train_data = qt.load(
                contextualized_embeddings=self._local_embeddings,
                bow_embeddings=train_bow,
                id2token=id2token)

            # Initialize FederatedCTM
            self.local_model = \
                FederatedCTM(self._model_parameters, self._grads_to_share, self._logger)

        else:
            self._logger.error("-- -- Provided underlying model not supported")

        # Initialize local_model with initial NN
        modelStateDict = proto_to_modelStateDict(
            response.initialNN.modelUpdate)
        optStateDict = proto_to_optStateDict(response.initialNN.optUpdate)
        self.local_model.model.load_state_dict(modelStateDict)
        self.local_model.optimizer.load_state_dict(optStateDict)

        self._logger.info(
            '-- -- Client %s initialized local model appropiately.', str(self.id))
        return

    def __send_ready_for_training(self) -> None:
        """Sends a message to the server to indicate that the client is ready to start the training phase.
        """

        self._logger.info(
            '-- -- Client %s sending ready for training phase.',
            str(self.id))

        # Generate request
        id_message = "ID" + str(self.id) + "_" + str(round(time.time()))
        header = federated_pb2.MessageHeader(
            id_request=id_message,
            message_type=federated_pb2.MessageType.CLIENT_READY_FOR_TRAINING)
        request = federated_pb2.ClientTensorRequest(header=header)

        # Send request to the server and wait for his response
        if self._stub:
            _ = self._stub.trainFederatedModel(request)

            self._logger.info(
                f"-- -- Client {self.id} stop behaving as client. Waiting for training to finish...")
            time.sleep(self._sleep_time)

        return
