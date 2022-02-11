# -*- coding: utf-8 -*-
"""
@author: lcalv
******************************************************************************
***                             CLASS CLIENT                               ***
******************************************************************************
"""
##############################################################################
#                                IMPORTS                                     #
##############################################################################
from __future__ import print_function
from importlib.metadata import metadata
import logging
import os
import time
import numpy as np
import datetime
import grpc
import multiprocessing as mp
import torch
from torch.utils.data import DataLoader
from gensim.corpora import Dictionary
import pathlib
from  gensim.test.utils import get_tmpfile

from avitm.avitm import AVITM
from federation import federated_pb2, federated_pb2_grpc
from utils.auxiliary_functions import get_file_chunks, save_chunks_to_file


##############################################################################
# DEBUG SETTINGS
##############################################################################
FORMAT = '[%(asctime)-15s] [%(filename)s] [%(levelname)s] %(message)s'
logging.basicConfig(format=FORMAT, level='INFO')
logger = logging.getLogger('client')


class Client:
    """ Class containing the stubs to interact with the server-side part via a gRPC channel.
    """

    def __init__(self, id, stub, period, local_corpus):
        self.id = id
        self.stub = stub
        self.period = period
        self.local_model = None
        self.local_corpus = local_corpus
        self.tmp_local_corpus = get_tmpfile(str(self.id))
        self.global_corpus = None
        self.tmp_global_corpus = get_tmpfile(str(self.id))

        # Save vocab in temporal local file
        self.__prepare_vocab_to_send(self.local_corpus)

        # Send file with vocabulary to server
        self.__send_local_vocab()

        # Wait for the consensed vocabulary
        self.__wait_for_agreed_vocab()

    def __prepare_vocab_to_send(self, corpus):
        dct = Dictionary(corpus)
        dct.save_as_text(self.tmp_local_corpus)

    def __send_local_vocab(self):
        print(self.tmp_local_corpus)
        print(type(self.tmp_local_corpus))
        request = get_file_chunks(self.tmp_local_corpus)
        
        # Send request to the server and wait for his response
        if self.stub:
            response = self.stub.upload(request)
            logger.info(
                'Client %s vocab is being sent to server.', str(self.id))
            #assert response.length == os.path.getsize(self.tmp_local_corpus)

            # Remove local file when finished the sending to the server
            if response.length == os.path.getsize(self.tmp_local_corpus):
                os.remove(self.tmp_local_corpus)

    def __wait_for_agreed_vocab(self):
        response = self.stub.download(federated_pb2.Empty())
        save_chunks_to_file(response, self.tmp_global_corpus)
        logger.info('Client %s receiving consensus vocab.', str(self.id))

    def __generate_protos_update(self, gradient):
        """Generates a prototocol buffer Update message from a Tensor gradient.

        Args:
        -----
           * gradient (torch.Tensor): Gradient to be sent in the protocol buffer message.

        Returns:
        --------
           * federated_pb2.Update: Prototocol buffer that is going to be send through the gRPC 
                                   channel.
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
        """Sends a minibatch's gradient update to the server.

        Args:
        -----
            * gradient (torch.Tensor): Gradient to be sent to the server in the current 
                                       minibatch.
            * current_mb (int):        Current minibatch, i.e. minibatch to which the gradient 
                                       is going to be sent corresponds.
            * current_epoch (int):     Current epoch, i.e. epoch to which the minibatch 
                                       corresponds.
            * num_epochs (int):        Number of epochs that is going to be used for training 
                                       the model.
        """
        # Generate request's header
        id_message = "ID" + str(self.id) + "_" + str(round(time.time()))
        header = federated_pb2.MessageHeader(id_request=id_message,
                                             message_type=federated_pb2.MessageType.CLIENT_TENSOR_SEND)
        # Generate request's metadata
        metadata = federated_pb2.MessageAdditionalData(current_mb=current_mb,
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
            logger.info('Client %s received a response to request %s',
                        str(self.id), response.header.id_to_request)

    def listen_for_updates(self):
        """Waits for an update from the server.

        Returns:
        --------
            * federated_pb2.ServerAggregatedTensorRequest: Update from the server with the  
                                                           average tensor generated from all federation clients' updates.
        """
        update = self.stub.sendAggregatedTensor(federated_pb2.Empty())
        logger.info('Client %s received updated for minibatch %s of epoch %s ',
                    str(self.id),
                    str(self.local_model.current_mb),
                    str(self.local_model.current_epoch))

        return update


"""
******************************************************************************
***                        CLASS AVITM CLIENT                              ***
******************************************************************************
"""


class AVITMClient(Client):

    def __init__(self, id, stub, period, local_corpus, model_parameters):

        Client.__init__(self, id, stub, period, local_corpus)

        self.model_parameters = model_parameters
        self.local_model = \
            AVITM(input_size=model_parameters["input_size"],
                  n_components=model_parameters["n_components"],
                  model_type=model_parameters["model_type"],
                  hidden_sizes=model_parameters["hidden_sizes"],
                  activation=model_parameters["activation"],
                  dropout=model_parameters["dropout"],
                  learn_priors=model_parameters["learn_priors"],
                  batch_size=model_parameters["batch_size"],
                  lr=model_parameters["lr"],
                  momentum=model_parameters["momentum"],
                  solver=model_parameters["solver"],
                  num_epochs=model_parameters["num_epochs"],
                  reduce_on_plateau=model_parameters["reduce_on_plateau"])

    def __train_epoch_local_model(self, loader):
        """Trains one epoch of the local AVITM model.

        Args:
        -----
            * loader (DataLoader): Python iterable over the training dataset with which the 
                                 minibatch is going to be trained.

        Returns:
        --------
            * int:   Number of samples processed
            * float: Minibatch's train loss

        """
        self.local_model.model.train()
        train_loss = 0
        samples_processed = 0

        self.local_model.current_mb = 0  # Counter for the current epoch
        for batch_samples in loader:

            # Training epoch starts
            X = batch_samples['X']
            if self.local_model.USE_CUDA:
                X = X.cuda()

            # Get gradients minibatch
            loss, train_loss, samples_processed = \
                self.local_model._train_minibatch(
                    X, train_loss, samples_processed)

            # Send minibatch' gradient to the server (gradient already converted to np)
            self.send_per_minibatch_gradient(
                self.local_model.model.prior_mean.grad.detach(),
                self.local_model.current_mb,
                self.local_model.current_epoch,
                self.local_model.num_epochs)

            logger.info('Client %s sent gradient %s/%s and is waiting for updates.',
                        str(self.id), str(self.local_model.current_mb),
                        str(self.local_model.current_epoch))

            # Wait until the server send the update
            request_update = self.listen_for_updates()

            # Update minibatch'gradient with the update from the server
            dims = tuple(
                [dim.size for dim in request_update.data.tensor_shape.dim])
            deserialized_bytes = np.frombuffer(
                request_update.data.tensor_content, dtype=np.float32)
            deserialized_numpy = np.reshape(
                deserialized_bytes, newshape=dims)
            deserialized_tensor = torch.Tensor(deserialized_numpy)

            # Calculate minibatch's train loss and samples processed
            train_loss, samples_processed = \
                self.local_model._optimize_on_minibatch(
                    X, loss, deserialized_tensor, train_loss, samples_processed)

            self.local_model.current_mb += 1  # One minibatch ends
        # One epoch ends

        # Calculate epoch's train loss and samples processed
        train_loss /= samples_processed

        return samples_processed, train_loss

    def train_local_model(self, train_dataset, save_dir=None):
        """Trains a local AVITM model. To do so, we send the server the gradients corresponding with the parameter beta of the model, and the server returns an update of such a parameter, which is calculated by averaging the per minibatch beta parameters that he gets from the set of clients that are connected to the federation. After obtaining the beta updates from the server, the client keeps with the training of its own local model. This process is repeated for each minibatch of each epoch.

        Args:
        -----
            * train_dataset (BOWDataset):        PyTorch Dataset classs for training data.
            * save_dir (pathlib.Path, optional): Directory to save checkpoint models to. 
                                                 Defaults to None.
        """
        # Print settings to output file
        print("Settings: \n\
               N Components: {}\n\
               Topic Prior Mean: {}\n\
               Topic Prior Variance: {}\n\
               Model Type: {}\n\
               Hidden Sizes: {}\n\
               Activation: {}\n\
               Dropout: {}\n\
               Learn Priors: {}\n\
               Learning Rate: {}\n\
               Momentum: {}\n\
               Reduce On Plateau: {}\n\
               Save Dir: {}".format(
            self.local_model.n_components, 0.0,
            1. - (1./self.local_model.n_components), self.local_model.model_type,
            self.local_model.hidden_sizes, self.local_model.activation,
            self.local_model.dropout, self.local_model.learn_priors,
            self.local_model.lr, self.local_model.momentum,
            self.local_model.reduce_on_plateau, save_dir))

        self.local_model.model_dir = save_dir
        self.local_model.train_data = train_dataset
        self.local_model.current_minibatch = 0
        self.local_model.current_epoch = 0

        # num_workers=mp.cpu_count()
        train_loader = DataLoader(
            self.local_model.train_data, batch_size=self.local_model.batch_size,
            shuffle=True, num_workers=0)

        # Initialize training variables
        train_loss = 0
        samples_processed = 0

        # Open channel for communication with the server
        #with grpc.insecure_channel('localhost:50051') as channel:
        #    self.stub = federated_pb2_grpc.FederationStub(channel)

        # Training of the local model
        for epoch in range(self.local_model.num_epochs):
            self.local_model.current_epoch = epoch

            # Train epoch
            s = datetime.datetime.now()
            sp, train_loss = self.__train_epoch_local_model(train_loader)
            samples_processed += sp
            e = datetime.datetime.now()

            # report
            print("Epoch: [{}/{}]\tSamples: [{}/{}]\tTrain Loss: {}\tTime: {}".format(
                epoch+1, self.local_model.num_epochs, samples_processed,
                len(self.local_model.train_data)*self.local_model.num_epochs, train_loss, e - s))

            # save best
            if train_loss < self.local_model.best_loss_train:
                self.local_model.best_loss_train = train_loss
                self.local_model.best_components = self.local_model.model.beta

                if save_dir is not None:
                    self.local_model.save(save_dir)


"""
******************************************************************************
***                        CLASS CTM CLIENT                                ***
******************************************************************************
"""


class CTMCLient(Client):
    def __init__(self, id, stub, period, local_corpus, model_parameters):

        Client.__init__(self, id, stub, period, local_corpus)

        self.model_parameters = model_parameters
        self.local_model = None
