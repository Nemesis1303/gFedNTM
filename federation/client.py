# -*- coding: utf-8 -*-
"""
@author: lcalv
******************************************************************************
***                             CLASS CLIENT                               ***
******************************************************************************
"""


import time

import numpy as np
from gensim.test.utils import get_tmpfile
from models.base.pytorchavitm.datasets.bow_dataset import BOWDataset
from models.federated.federated_avitm import FederatedAVITM
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import FeatureUnion
from utils.auxiliary_functions import (proto_to_modelStateDict,
                                       proto_to_optStateDict, serializeTensor)

from federation import federated_pb2


class Client:
    """ Class containing the stubs to interact with the server-side part via a gRPC channel.
    """

    def __init__(self, id, stub, local_model_type, local_corpus, model_parameters, logger=None):
        """
        Sets the main attributes defining a Client

        Parameters
        ----------
        id : int
            Client's ide
        stub : federated_pb2_grpc.FederationStub
            Module acting as the interface for gRPC client
        local_model_type : str
            Type of the underlying topic modeling algorithm used for the federated training
        local_corpus : List[str]
            List of documents that constitute the node's local corpus
        """

        self.id = id
        self.stub = stub
        self.local_model_type = local_model_type
        self.local_corpus = \
            [" ".join(local_corpus[i]) for i in np.arange(len(local_corpus))]
        self.model_parameters = model_parameters

        # Other attributes
        self.local_model = None
        self.tmp_local_vocab = get_tmpfile(str(self.id))
        self.tmp_global_vocab = get_tmpfile(str(self.id))
        self.train_data = None

        # Create logger object
        if logger:
            self.logger = logger
        else:
            import logging
            FMT = '[%(asctime)-15s] [%(filename)s] [%(levelname)s] %(message)s'
            logging.basicConfig(format=FMT, level='INFO')
            self.logger = logging.getLogger('Client')

        # Send file with vocabulary to server
        self.__send_local_vocab()

        # Wait for the consensed vocabulary and initial NN
        self.__wait_for_agreed_vocab_NN()

    def __prepare_vocab_to_send(self):
        """
        Gets the vocabulary associated to the local corpus as a dictionary object.
        """

        if self.local_model_type == 'prod':
            # Create a CountVectorizer object to convert a collection of text documents into a matrix of token counts
            cv = CountVectorizer(
                input='content', lowercase=True, stop_words='english', binary=False)
            # Learn the vocabulary dictionary, bow = document-term matrix
            cv.fit_transform(self.local_corpus)
            vocab_dict = cv.vocabulary_
        elif self.local_model_type == 'ctm':
            print("To be implemented")
        else:
            self.logger.error(
                "Model type for the vocabulary object generation not supported")

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
        if self.stub:
            response = self.stub.sendLocalDic(dic)
            self.logger.info(
                'Client %s vocab is being sent to server.', str(self.id))
            # Remove local file when finished the sending to the server
            if response.length == len(vocab_dict):
                self.logger.info(
                    'Server received correctly vocab from client %s.', str(self.id))

        return

    def __wait_for_agreed_vocab_NN(self):
        """
        Waits while receiving the agreed vocabulary sent by the server.
        """

        self.logger.info(
            'Client %s receiving consensus vocab and initial NN.', str(self.id))

        # Get global dictionary
        response = self.stub.sendGlobalDicAndInitialNN(federated_pb2.Empty())

        # Unprotofy global dictionary
        vocabs = []
        for dic_i in range(len(response.dic)):
            vocab_i = dict([(pair.key, pair.value.ivalue)
                            for pair in response.dic[dic_i].pairs])
            cv_i = CountVectorizer(vocabulary=vocab_i)
            name_i = "CV" + str(dic_i)
            vocabs.append((name_i, cv_i))

        self.global_vocab = FeatureUnion(vocabs)

        # Initialize local_model with initial NN
        modelStateDict = proto_to_modelStateDict(
            response.initialNN.modelUpdate)
        optStateDict = proto_to_optStateDict(response.initialNN.optUpdate)

        if self.local_model_type == "prod":

            # Local document-term matrix as function of the global vocabulary
            train_bow = self.global_vocab.transform(
                self.local_corpus).toarray()

            # Array mapping from feature integer indices to feature name
            idx2token = self.global_vocab.get_feature_names_out()
            self.model_parameters["input_size"] = len(idx2token)
            self.model_parameters["id2token"] = \
                {k: v for k, v in zip(range(0, len(idx2token)), idx2token)}

            # The train dataset is an object from the class BOWDataset
            self.train_data = BOWDataset(train_bow, idx2token)

            self.local_model = \
                FederatedAVITM(self.model_parameters, self, self.logger)

        elif self.local_model_type == "ctm":
            print("To be implemented")
        else:
            self.logger.error("Provided underlying model not supported")

        self.local_model.model.load_state_dict(modelStateDict)
        self.local_model.optimizer.load_state_dict(optStateDict)

        self.logger.info(
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
        if self.stub:
            response = self.stub.sendLocalTensor(request)
            self.logger.info('Client %s received a response to request %s',
                             str(self.id), response.header.id_to_request)

    def listen_for_updates(self):
        """
        Waits for an update from the server.

        Returns
        -------
        update : federated_pb2.ServerAggregatedTensorRequest
            Update from the server with the average tensor generated from all federation clients' updates.
        """

        update = self.stub.sendAggregatedTensor(federated_pb2.Empty())
        self.logger.info('Client %s received updated for minibatch %s of epoch %s ',
                         str(self.id),
                         str(self.local_model.current_mb),
                         str(self.local_model.current_epoch))

        return update

    def train_local_model(self):
        """
        Trains a the local model. 

        To do so, we send the server the gradients corresponding with the parameter beta of the model, and the server returns an update of such a parameter, which is calculated by averaging the per minibatch beta parameters that he gets from the set of clients that are connected to the federation. After obtaining the beta updates from the server, the client keeps with the training of its own local model. This process is repeated for each minibatch of each epoch.
        """

        self.local_model.fit(self.train_data)

        print("TRAINED")

        return
    
    def eval_local_model(self, eval_params):
        self.local_model.get_results_model()
        self.local_model.evaluate_synthetic_model(
            eval_params[0], eval_params[1], eval_params[2])
        print("EVALUATED")
