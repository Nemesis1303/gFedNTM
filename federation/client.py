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
import sys
import getopt
import logging
import time
import numpy as np
from timeloop import Timeloop
import datetime
import grpc
import multiprocessing as mp
import torch
from torch.utils.data import DataLoader

from avitm.avitm import AVITM
from federation import federated_pb2, federated_pb2_grpc


##############################################################################
# DEBUG SETTINGS
##############################################################################
FORMAT = '[%(asctime)-15s] [%(filename)s] [%(levelname)s] %(message)s'
logging.basicConfig(format=FORMAT, level='INFO')
logger = logging.getLogger('client')


class Client:
    """ Class containing the stubs to interact with the server-side part via a gRPC channel.
    """

    def __init__(self, id, period, model_parameters):
        self.id = id
        self.period = period
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
    
    def __train_epoch_local_model(self,loader):
        self.local_model.model.train()
        train_loss = 0
        samples_processed = 0

        self.local_model.current_mb = 0 # Counter for the current epoch
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
            self.__send_per_minibatch_gradient(
                self.local_model.model.prior_mean.grad.detach(),
                self.local_model.current_mb,
                self.local_model.current_epoch,
                self.local_model.num_epochs)

            print("Client ", self.id, "sent gradient ", self.local_model.current_mb,
                    "/", self.local_model.current_epoch, "and is waiting for updates")

            # Wait until the server send the update
            request_update = self.__listen_for_updates()

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

            self.local_model.current_mb += 1 # One minibatch ends
        # One epoch ends

        # Calculate epoch's train loss and samples processed
        train_loss /= samples_processed

        return samples_processed, train_loss


    def train_local_model(self, train_dataset, save_dir=None):
        """
        Train a local AVITM model. To do so, we send the server the gradients corresponding with the parameter beta of the model, and the server returns an update of such a parameter, which is calculated by averaging the per minibatch beta parameters that he gets from the set of clients that are connected to the federation. After obtaining the beta updates from the server, the client keeps with the training of its own local model. This process is repeated for each minibatch of each epoch.

        Args
            * train_dataset : PyTorch Dataset classs for training data.
            * save_dir : directory to save checkpoint models to.
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
        with grpc.insecure_channel('localhost:50051') as channel:
            self.stub = federated_pb2_grpc.FederationStub(channel)

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

    def __send_per_minibatch_gradient(self, gradient, current_mb, current_epoch, num_epochs):
        id_message = "ID" + str(self.id) + "_" + str(round(time.time()))
        header = federated_pb2.MessageHeader(id_request=id_message,
                                             message_type=federated_pb2.MessageType.CLIENT_TENSOR_SEND)
        metadata = federated_pb2.MessageAdditionalData(current_mb=current_mb,
                                                       current_epoch=current_epoch,
                                                       num_max_epochs=num_epochs,
                                                       id_machine=int(self.id))
        update_name = "Update of " + str(self.id)
        content_bytes = gradient.numpy().tobytes()
        content_type = gradient.numpy().dtype
        size = federated_pb2.TensorShape()
        # size.dim.extend([federated_pb2.TensorShape.Dim(size=gradient.size(dim=0), name="dim1"),
        #                 federated_pb2.TensorShape.Dim(size=gradient.size(dim=1), name="dim2")])
        size.dim.extend([federated_pb2.TensorShape.Dim(
            size=gradient.size(dim=0), name="dim1")])
        # TODO: Figure out how to write dtype=content_type
        data = federated_pb2.Update(tensor_name=update_name,
                                    tensor_shape=size,
                                    tensor_content=content_bytes)
        request = federated_pb2.ClientTensorRequest(
            header=header, metadata=metadata, data=data)
        if self.stub:
            response = self.stub.sendLocalTensor(request)
            logger.info('Client %s received a response to request %s',
                        str(self.id), response.header.id_to_request)

    def __listen_for_updates(self):
        update = self.stub.sendAggregatedTensor(federated_pb2.Empty())
        print("Client with ID ", self.id,
              "recevied update for minibatch ", self.local_model.current_mb, " of the epoch ", self.local_model.current_epoch)
        return update
