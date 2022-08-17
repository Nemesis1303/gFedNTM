# -*- coding: utf-8 -*-
"""
@author: lcalv
******************************************************************************
***                             CLASS CLIENT                               ***
******************************************************************************
"""

from __future__ import print_function

import os
import time

import numpy as np
from gensim.test.utils import get_tmpfile
from models.federated.federated_avitm import SyntheticFederatedAVITM
from utils.auxiliary_functions import (get_corpus_from_file, get_file_chunks,
                                       save_chunks_to_file,
                                       save_corpus_in_file)
from utils.utils_preprocessing import prepare_data_avitm_federated

from federation import federated_pb2


class Client:
    """ Class containing the stubs to interact with the server-side part via a gRPC channel.
    """

    def __init__(self, id, stub, period, local_corpus, model_parameters, logger=None):
        self.id = id
        self.stub = stub
        self.period = period
        self.local_model = None
        self.model_parameters = model_parameters
        self.local_corpus = local_corpus
        self.tmp_local_corpus = get_tmpfile(str(self.id))
        self.global_corpus = None
        self.tmp_global_corpus = get_tmpfile(str(self.id))

        # Create logger object
        if logger:
            self.logger = logger
        else:
            import logging
            FMT = '[%(asctime)-15s] [%(filename)s] [%(levelname)s] %(message)s'
            logging.basicConfig(format=FMT, level='INFO')
            self.logger = logging.getLogger('Client')

        # Save vocab in temporal local file
        self.__prepare_vocab_to_send(self.local_corpus)

        # Send file with vocabulary to server
        self.__send_local_vocab()

        # Wait for the consensed vocabulary
        self.__wait_for_agreed_vocab()

    def __prepare_vocab_to_send(self, corpus):
        """
        Prepares the vocabulary that a node is going to send to the server by saving it into a text file.

        Parameters
        ----------
        corpus : numpy.ndarray
            Node's corpus
        """
        save_corpus_in_file(corpus, self.tmp_local_corpus)

    def __send_local_vocab(self):
        """
        Sends the local vocabulary to the server and waits for its ACK.
        """
        request = get_file_chunks(self.tmp_local_corpus)

        # Send request to the server and wait for his response
        if self.stub:
            response = self.stub.upload(request)
            self.logger.info(
                'Client %s vocab is being sent to server.', str(self.id))
            # Remove local file when finished the sending to the server
            if response.length == os.path.getsize(self.tmp_local_corpus):
                os.remove(self.tmp_local_corpus)

    def __wait_for_agreed_vocab(self):
        """
        Waits by saving the agreed vocabulary sent by the server.
        """
        response = self.stub.download(federated_pb2.Empty())
        save_chunks_to_file(response, self.tmp_global_corpus)
        self.logger.info('Client %s receiving consensus vocab.', str(self.id))
        self.global_corpus = get_corpus_from_file(self.tmp_global_corpus)

    def __generate_protos_update(self, gradient):
        """
        Generates a prototocol buffer Update message from a Tensor gradient.

        Parameters
        ----------
        gradient : torch.Tensor
            Gradient to be sent in the protocol buffer message

        Returns
        -------
        data : federated_pb2.Update
            Prototocol buffer that is going to be send through the gRPC channel
        """

        # Name of the update based on client's id
        update_name = "Update from " + str(self.id)

        # Convert Tensor gradient to bytes object for sending
        content_bytes = gradient.numpy().tobytes()
        content_type = str(gradient.numpy().dtype)
        size = federated_pb2.TensorShape()
        num_dims = len(gradient.shape)
        for i in np.arange(num_dims):
            name = "dim" + str(i)
            size.dim.extend(
                [federated_pb2.TensorShape.Dim(size=gradient.shape[i], name=name)])
        data = federated_pb2.Update(tensor_name=update_name,
                                    dtype=content_type,
                                    tensor_shape=size,
                                    tensor_content=content_bytes)
        return data

    def send_per_minibatch_gradient(self, gradient, current_mb, current_epoch, num_epochs):
        """
        Sends a minibatch's gradient update to the server.

        Parameters
        ----------
        gradient : torch.Tensor
            Gradient to be sent to the server in the current minibatch
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
        # Generate Protos Update
        data = self.__generate_protos_update(gradient)

        # Generate Protos ClientTensorRequest with the update
        request = federated_pb2.ClientTensorRequest(
            header=header, metadata=metadata, data=data)

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

        # Create local model
        # TODO: Create model according to type

        # Generate training dataset in the format for AVITM
        self.train_data, input_size, id2token = \
            prepare_data_avitm_federated(
                self.global_corpus, 0.99, 0.01)

        self.model_parameters["input_size"] = input_size
        self.model_parameters["id2token"] = id2token

        self.local_model = \
            SyntheticFederatedAVITM(self.model_parameters, self, self.logger)
        self.local_model.fit(self.train_data)
        print("TRAINED")

    def eval_local_model(self, eval_params):
        self.local_model.get_results_model()
        self.local_model.evaluate_synthetic_model(
            eval_params[0], eval_params[1], eval_params[2], eval_params[3])
        print("EVALUATED")
