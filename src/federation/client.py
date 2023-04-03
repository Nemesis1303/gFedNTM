# -*- coding: utf-8 -*-
"""
Created on Feb 1, 2022

@author: L. Calvo-Bartolomé (lcalvo@pa.uc3m.es)
"""
import time

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from src.models.base.contextualized_topic_models.utils.data_preparation import \
    TopicModelDataPreparation
from src.models.base.pytorchavitm.datasets.bow_dataset import BOWDataset
from src.models.federated.federated_avitm import FederatedAVITM
from src.models.federated.federated_ctm import FederatedCTM
from src.protos import federated_pb2, federated_pb2_grpc
from src.utils.auxiliary_functions import (proto_to_modelStateDict,
                                           proto_to_optStateDict,
                                           serializeTensor)


class Client:
    """ Class containing the stubs to interact with the server-side part via a gRPC channel.
    """

    def __init__(self,
                 id: int,
                 stub: federated_pb2_grpc.FederationStub,
                 local_corpus: list,
                 data_type: str,
                 field_mappings: dict,
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
        self._local_corpus, self._local_embeddings = \
            self.__get_local_corpus(data_type, local_corpus, field_mappings)

        # Other attributes
        self._local_model_type = None
        self._model_parameters = None
        self._local_model = None
        self._train_data = None

        # Create logger object
        if logger:
            self._logger = logger
        else:
            import logging
            FMT = '[%(asctime)-15s] [%(filename)s] [%(levelname)s] %(message)s'
            logging.basicConfig(format=FMT, level='INFO',
                                filename="logs_client.txt")
            self._logger = logging.getLogger('Client')

        # Send vocabulary to server
        self.__send_local_vocab()

        # Wait for the consensed vocabulary and initial NN
        self.__wait_for_agreed_vocab_NN()

    def __get_local_corpus(self, data_type, corpus, field_mappings):
        """
        Gets the the local training corpus based on whether the input provided is synthetic or real
        """

        local_embeddings = None
        if data_type == "synthetic":
            local_corpus = \
                [" ".join(corpus[i]) for i in np.arange(len(corpus))]
        else:
            text_field = field_mappings['raw_text']
            emb_field = field_mappings['embeddings']
            df_lemas = corpus[[text_field]].values.tolist()
            local_corpus = [' '.join(doc) for doc in df_lemas]
            if emb_field in list(corpus.columns.values):
                local_embeddings = corpus[emb_field].values

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

        # Send dictionary to the server and wait for his response
        if self._stub:
            response = self._stub.sendLocalDic(dic)
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
            self._train_data = BOWDataset(train_bow, idx2token)

            # Initialize FederatedAVITM
            self._local_model = \
                FederatedAVITM(self._model_parameters, self, self._logger)

        elif self._local_model_type == "ctm":

            # The train dataset is an object from the class CTMDataset
            qt = TopicModelDataPreparation()
            qt.vectorizer = self._global_vocab
            qt.id2token = id2token
            qt.vocab = idx2token
            self._train_data = qt.load(
                contextualized_embeddings=self._local_embeddings,
                bow_embeddings=train_bow,
                id2token=id2token)

            # Initialize FederatedCTM
            self._local_model = \
                FederatedCTM(self._model_parameters, self, self._logger)

        else:
            self._logger.error("Provided underlying model not supported")

        # Initialize local_model with initial NN
        modelStateDict = proto_to_modelStateDict(
            response.initialNN.modelUpdate)
        optStateDict = proto_to_optStateDict(response.initialNN.optUpdate)
        self._local_model.model.load_state_dict(modelStateDict)
        self._local_model.optimizer.load_state_dict(optStateDict)

        self._logger.info(
            'Client %s initialized local model appropiately.', str(self.id))

        return

    def send_per_minibatch_gradient(self, gradients, current_mb, current_epoch, num_epochs):
        """
        Sends a minibatch's gradient update to the server.

        Parameters
        ----------
        gradients : List[List[gradient_name,gradient_value]]
            Gradients to be sent to the server in the current minibatch
        current_mb: int
            Current minibatch, i.e. minibatch to which the gradient that is going to be sent corresponds
        current_epoch: int
            Current epoch, i.e. epoch to which the minibatch corresponds
        num_epochs: int
            Number of epochs that is going to be used for training the model.

        Returns
        -------
        data : federated_pb2.Update
            Prototocol buffer that is going to be send through the gRPC channel
        """

        # Generate request's header
        id_message = "ID" + str(self.id) + "_" + str(round(time.time()))
        header = federated_pb2.MessageHeader(id_request=id_message,
                                             message_type=federated_pb2.MessageType.CLIENT_TENSOR_SEND)
        # Generate request's metadata
        metadata = \
            federated_pb2.MessageAdditionalData(current_mb=current_mb,
                                                current_epoch=current_epoch,
                                                num_max_epochs=num_epochs,
                                                id_machine=int(self.id))
        # Generate Protos Updates
        updates_ = []
        for gradient in gradients:
            tensor_protos = serializeTensor(gradient[1])
            protos_update = federated_pb2.Update(
                tensor_name=gradient[0],
                tensor=tensor_protos
            )
            updates_.append(protos_update)

        # Generate Protos ClientTensorRequest with the update
        request = federated_pb2.ClientTensorRequest(
            header=header, metadata=metadata, updates=updates_)

        # Send request to the server and wait for his response
        if self._stub:
            response = self._stub.sendLocalTensor(request)
            self._logger.info('Client %s received a response to request %s',
                              str(self.id), response.header.id_to_request)

    def listen_for_updates(self):
        """
        Waits for an update from the server.

        Returns
        -------
        update : federated_pb2.ServerAggregatedTensorRequest
            Update from the server with the average tensor generated from all federation clients' updates.
        """

        update = self._stub.sendAggregatedTensor(federated_pb2.Empty())
        self._logger.info('Client %s received updated for minibatch %s of epoch %s ',
                          str(self.id),
                          str(self._local_model.current_mb),
                          str(self._local_model.current_epoch))

        return update

    def train_local_model(self):
        """
        Trains a the local model. 

        To do so, we send the server the gradients corresponding with the parameter beta of the model, and the server returns an update of such a parameter, which is calculated by averaging the per minibatch beta parameters that he gets from the set of clients that are connected to the federation. After obtaining the beta updates from the server, the client keeps with the training of its own local model. This process is repeated for each minibatch of each epoch.
        """

        self._local_model.fit(self._train_data)
        self._local_model.get_results_model()

        return

    def eval_local_model(self, eval_params):
        """
        Evaluates the local model if synthetic data is being used.
        """

        self._local_model.get_results_model()
        self._local_model.evaluate_synthetic_model(
            eval_params[0], eval_params[1], eval_params[2])
