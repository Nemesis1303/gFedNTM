# -*- coding: utf-8 -*-
"""
This script contains the two classes that define the client-side part of the federated learning process. 

* The CLIENT class: This class contains stubs that facilitate interactions with the server-side via a gRPC channel during the vocabulary consensus phase.

* The FEDERATEDCLIENTSERVER class: This class defines the behavior of the client when it acts as a 'client-server' to process server petitions for another gradient update. T

Created on Feb 1, 2022
Last updated on Jul 29, 2023
@author: L. Calvo-Bartolomé (lcalvo@pa.uc3m.es)
"""
import time
from typing import Union
import threading
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
GRPC_TRACE=all

# ======================================================
# CLIENT-SERVER
# ======================================================


class FederatedClientServer(federated_pb2_grpc.FederationServerServicer):
    """Class that describes the behaviour to be followed by the client when it acts as a 'client-server' to process the server petitions.
    """

    def __init__(self,
                 local_model: Union[FederatedAVITM, FederatedCTM],
                 train_data: Union[BOWDataset, CTMDataset],
                 id: int,
                 logger=None):

        self._local_model = local_model
        self.id = id
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

    def getGradient(self,
                    request: federated_pb2.ServerGetGradientRequest,
                    context: grpc.AuthMetadataContext) -> federated_pb2.ClientTensorRequest:
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
        # TODO: See what to do with this
        self.global_epoch = request.iter
      
        # Generate gradient update (train one mb of local model)
        gr_upt = self._local_model.train_mb_delta()
        gradients_ = []
        for key in gr_upt.keys():
            if key not in ["current_mb", "current_epoch", "num_epochs"]:
                gradients_.append([key, gr_upt[key].grad.detach()])

        # Generate response's header
        id_message = "ID" + str(self.id) + "_" + str(round(time.time()))
        header = federated_pb2.MessageHeader(
            id_request=id_message,
            message_type=federated_pb2.MessageType.CLIENT_TENSOR_SEND)

        # Generate response's metadata
        metadata = \
            federated_pb2.MessageAdditionalData(
                current_mb=gr_upt["current_mb"],
                current_epoch=gr_upt["current_epoch"],
                num_max_epochs=gr_upt["num_epochs"],
                id_machine=int(self.id))

        # Generate Protos Updates
        updates_ = []
        for gradient in gradients_:
            tensor_protos = serializeTensor(gradient[1])
            protos_update = federated_pb2.Update(
                tensor_name=gradient[0],
                tensor=tensor_protos
            )
            updates_.append(protos_update)

        # Generate Protos ClientTensorRequest with the update
        response = federated_pb2.ClientTensorRequest(
            header=header, metadata=metadata, updates=updates_)

        self._logger.info('Client %s sent update for global epoch %s',
                          str(self.id), self.global_epoch)

        return response

    def sendAggregatedTensor(self,
                             request: federated_pb2.ServerAggregatedTensorRequest,
                             context: grpc.AuthMetadataContext) -> federated_pb2.ClientReceivedResponse:
        """Process the receival of the aggregated tensor from the server and generates and ACK message.

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

        # Process request and update local model with server's weights
        modelStateDict = proto_to_modelStateDict(
            request.nndata.modelUpdate)

        # TODO: Check if this is needed
        optStateDict = proto_to_optStateDict(request.nndata.optUpdate)
        self._local_model.deltaUpdateFit(modelStateDict)

        header = federated_pb2.MessageHeader(
            id_request=str(self.global_epoch),
            message_type=federated_pb2.MessageType.CLIENT_CONFIRM_RECEIVED)
        response = federated_pb2.ClientReceivedResponse()

        return response

# ======================================================
# CLIENT
# ======================================================
class Client:
    """ Class that describes the client's behaviour during the vocabulary consensus phase.
    """

    def __init__(self,
                 id: int,
                 stub: federated_pb2_grpc.FederationStub,
                 local_corpus: list,
                 data_type: str,
                 opts_server: dict,
                 logger=None):
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
        """

        self.id = id
        self._stub = stub

        # Create logger object
        if logger:
            self._logger = logger
        else:
            import logging
            FMT = '[%(asctime)-15s] [%(filename)s] [%(levelname)s] %(message)s'
            logging.basicConfig(format=FMT, level='DEBUG')
            self._logger = logging.getLogger('Client')
            self._logger.setLevel(logging.DEBUG)
        
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
        self.__start_client_server(opts_server)

        # Send ready for training after vocabulary consensus phase
        self.__send_ready_for_training()

    def __start_client_server(self, opts_server: dict):
        client_server = grpc.server(futures.ThreadPoolExecutor(), options=opts_server)
        federated_server = FederatedClientServer(
            self.local_model, self.train_data, self.id)
        federated_pb2_grpc.add_FederationServerServicer_to_server(
            federated_server, client_server)
        client_server.add_insecure_port('[::]:' + str(50051 + self.id))
        client_server.start()
        federated_server._logger.info(f"Client Server started at {str(50051 + self.id)}")
        try:
            thread = threading.Thread(target=client_server.wait_for_termination)
            thread.daemon = True
            thread.start()
            self._logger.info("Client-server thread started")
        except:
            self._logger.info("Could not start client-server thread")
        #client_server.wait_for_termination()
        #federated_server._logger.info("Client Server TERMINATED")
        return

    def __get_local_corpus(self, data_type, corpus):
        """
        Gets the the local training corpus based on whether the input provided is synthetic or real
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
        return local_corpus, local_embeddings

    def __prepare_vocab_to_send(self):
        """
        Gets the vocabulary associated to the local corpus as a dictionary object.
        """

        # Create a CountVectorizer object to convert a collection of text documents into a matrix of token counts
        cv = CountVectorizer(
            input='content', lowercase=True, stop_words='english', binary=False)
        # Learn the vocabulary dictionary, bow = document-term matrix
        cv.fit_transform(self._local_corpus)
        vocab_dict = cv.vocabulary_

        return vocab_dict

    def __send_local_vocab(self):
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
                'Client %s vocab is being sent to server.', str(self.id))
            if response.length == len(vocab_dict):
                self._logger.info(
                    'Server received correctly vocab from client %s.', str(self.id))

        return

    def __wait_for_agreed_vocab_NN(self):
        """
        Waits while receiving the agreed vocabulary sent by the server.
        """

        self._logger.info(
            'Client %s receiving consensus vocab and initial NN.',
            str(self.id))

        # Get global model_parameters, model_type, dictionary and initialized NN
        response = self._stub.sendGlobalDicAndInitialNN(federated_pb2.Empty())

        self._logger.info(
            'Client %s response received.',
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
            'Client %s received model params and model type.',
            str(self.id))

        # Unprotofy global dictionary
        vocabs = []
        for dic_i in range(len(response.dic)):
            vocab_i = dict([(pair.key, pair.value.ivalue)
                            for pair in response.dic[dic_i].pairs])
            vocabs.append(vocab_i)

        self._global_vocab = CountVectorizer(vocabulary=vocabs[0])
        self._logger.info(
            'Client %s model vocab unprotofied.',
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
                FederatedAVITM(self._model_parameters, self._logger)

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
                FederatedCTM(self._model_parameters, self, self._logger)

        else:
            self._logger.error("Provided underlying model not supported")

        # Initialize local_model with initial NN
        modelStateDict = proto_to_modelStateDict(
            response.initialNN.modelUpdate)
        optStateDict = proto_to_optStateDict(response.initialNN.optUpdate)
        self.local_model.model.load_state_dict(modelStateDict)
        self.local_model.optimizer.load_state_dict(optStateDict)

        self._logger.info(
            'Client %s initialized local model appropiately.', str(self.id))

        return

    def __send_ready_for_training(self):

        self._logger.info(
            'Client %s sending ready for training phase.',
            str(self.id))

        # Generate request
        id_message = "ID" + str(self.id) + "_" + str(round(time.time()))
        header = federated_pb2.MessageHeader(
            id_request=id_message,
            message_type=federated_pb2.MessageType.CLIENT_READY_FOR_TRAINING)
        request = federated_pb2.ClientTensorRequest(header=header)

        # Send request to the server and wait for his response
        if self._stub:
            response = self._stub.trainFederatedModel(request)

        return