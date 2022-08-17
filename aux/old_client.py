# -*- coding: utf-8 -*-
"""
@author: lcalv
******************************************************************************
***                         CLASS OLD_CLIENT                               ***
******************************************************************************
"""
"""To run in main:

client = AVITMClient(id_client, stub, period, corpus, model_parameters) 
client = SyntheticAVITMClient(id_client, stub, period, corpus, model_parameters, vocab_size, doc_topic_distrib_gt_all[id_client-1], word_topic_distrib_gt, file_save)
"""
##############################################################################
#                                IMPORTS                                     #
##############################################################################
from __future__ import print_function

import datetime
import os
import time

import numpy as np
import torch
from gensim.test.utils import get_tmpfile
from models.federated.federated_avitm import SyntheticFederatedAVITM
from models.pytorchavitm.federated.federated_avitm import FederatedAVITM
from torch.utils.data import DataLoader
from utils.auxiliary_functions import (get_corpus_from_file, get_file_chunks,
                                       save_chunks_to_file,
                                       save_corpus_in_file, save_model_as_npz)
from utils.utils_postprocessing import (convert_topic_word_to_init_size,
                                        thetas2sparse)
from utils.utils_preprocessing import prepare_data_avitm_federated

from federation import federated_pb2

class Client:
    """ Class containing the stubs to interact with the server-side part via a gRPC channel.
    """

    def __init__(self, id, stub, period, local_corpus, model_parameters,logger=None):
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
        print("HERE")
        print(type(self.model_parameters["input_size"]))
        self.model_parameters["id2token"] = id2token

        self.local_model = \
            SyntheticFederatedAVITM(self.model_parameters, self, self.logger)
        self.local_model.fit(self.train_data)
        print("TRAINED")
    
    def eval_local_model(self, eval_params):
        self.local_model.get_results_model()
        self.local_model.evaluate_synthetic_model(eval_params[0], eval_params[1], eval_params[2], eval_params[3])
        print("EVALUATED")


"""
******************************************************************************
***                        CLASS AVITM CLIENT                              ***
******************************************************************************
"""
class AVITMClient(Client):

    def __init__(self, id, stub, period, local_corpus, model_parameters):

        Client.__init__(self, id, stub, period, local_corpus)

        self.model_parameters = model_parameters

        # Generate training set
        self.train_dataset = None
        self.input_size = None
        self.id2token = None
        self.__get_training_dataset()

        # Configure local model
        self.local_model = \
            FederatedAVITM(logger=self.logger,
                           input_size=self.input_size,
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
                           reduce_on_plateau=model_parameters["reduce_on_plateau"],
                           num_samples=model_parameters["num_samples"],
                           num_data_loader_workers=model_parameters["num_data_loader_workers"],
                           verbose=True)

        # Start training
        self.__train_local_model(self.train_dataset)

        # Get topics, doc-topic and word-topic distributions
        self.__get_results_model()

    def __get_training_dataset(self):
        # Generate training dataset in the format for AVITM
        self.train_dataset, self.input_size, self.id2token = \
            prepare_data_avitm_federated(self.global_corpus, 0.99, 0.01)


    def __train_epoch_local_model(self, loader):
        """
        Trains one epoch of the local AVITM model.

        Parameters
        ----------
        loader: DataLoader
            Python iterable over the training dataset with which the epoch is going to be trained.

        Returns
        -------
        samples_processed: int
            Number of processed samples
        train_loss: float
            Training loss
        """

        self.local_model.model.train()

        train_loss = 0
        samples_processed = 0

        # Counter for the current minibatch
        self.local_model.current_mb = 0  

        # Training epoch starts
        for batch_samples in loader:
            
            # Get samples in minibatch
            X = batch_samples['X']

            # Get gradients minibatch
            loss, train_loss, samples_processed = \
                self.local_model._train_minibatch(
                    X, train_loss, samples_processed)

            # Send minibatch' gradient to the server (gradient already converted to np)
            self.send_per_minibatch_gradient(
                self.local_model.model.beta.grad.detach(),
                self.local_model.current_mb,
                self.local_model.current_epoch,
                self.local_model.num_epochs)

            self.logger.info('Client %s sent gradient %s/%s and is waiting for updates.',
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
        # Training epoch ends

        # Calculate epoch's train loss and samples processed
        train_loss /= samples_processed

        return samples_processed, train_loss

    def __train_local_model(self, train_dataset, save_dir=None):
        """
        Trains a local AVITM model. 
        
        To do so, we send the server the gradients corresponding with the parameter beta of the model, and the server returns an update of such a parameter, which is calculated by averaging the per minibatch beta parameters that he gets from the set of clients that are connected to the federation. After obtaining the beta updates from the server, the client keeps with the training of its own local model. This process is repeated for each minibatch of each epoch.

        Parameters
        ----------
        train_dataset: BOWDataset
            PyTorch Dataset classs for training data
        save_dir: pathlib.Path, optional
            Directory to save checkpoint models to, defaults to None.
        """
        
        self.local_model.model_dir = save_dir
        self.local_model.train_data = train_dataset
        # TODO: Include validation data
        # self.local_model.validation_data = validation_data
        self.local_model.current_minibatch = 0
        self.local_model.current_epoch = 0

        # num_workers=mp.cpu_count()
        train_loader = DataLoader(
            self.local_model.train_data, batch_size=self.local_model.batch_size,
            shuffle=True, num_workers=0)

        # Initialize training variables
        train_loss = 0
        samples_processed = 0

        # Training of the local model
        for epoch in range(self.local_model.num_epochs):
            self.local_model.current_epoch = epoch

            # Train epoch
            s = datetime.datetime.now()
            sp, train_loss = \
                self.__train_epoch_local_model(train_loader)
            samples_processed += sp
            e = datetime.datetime.now()

            # report
            print("Epoch: [{}/{}]\tSamples: [{}/{}]\tTrain Loss: {}\tTime: {}".format(
                epoch+1, self.local_model.num_epochs, samples_processed,
                len(self.local_model.train_data)*self.local_model.num_epochs, train_loss, e - s))

            # save best
            if train_loss < self.local_model.best_loss_train:
                self.local_model.best_components = self.local_model.model.beta
                self.local_model.best_loss_train = train_loss

                if save_dir is not None:
                    self.local_model.save(save_dir)

    def __get_results_model(self):

        # Get topics
        self.local_model.topics = self.local_model.get_topics() 
        #pd.DataFrame(self.local_model.get_topics()).T

        # Get doc-topic distribution
        self.local_model.doc_topic_distrib = \
            np.asarray(self.local_model.get_doc_topic_distribution(self.local_model.train_data))#.T 
        #print(self.local_model.doc_topic_distrib == self.local_model.get)        

        # Get word-topic distribution
        self.local_model.word_topic_distrib = self.local_model.get_topic_word_distribution()
        

"""
******************************************************************************
***                     CLASS SYNTHETIC AVITM CLIENT                       ***
******************************************************************************
"""
class SyntheticAVITMClient(AVITMClient):
    def __init__(self, id, stub, period, local_corpus, model_parameters, vocab_size,
                 gt_doc_topic_distrib, gt_word_topic_distrib, file_save):

        AVITMClient.__init__(self, id, stub, period,
                             local_corpus, model_parameters)

        self.vocab_size = vocab_size
        self.gt_doc_topic_distrib = gt_doc_topic_distrib
        self.gt_word_topic_distrib = gt_word_topic_distrib

        self.sim_mat_thetas_gt = None
        self.sim_mat_thetas_model = None
        self.sim_mat_betas = None
        self.sim_docs_frob = 0.0
        self.sim_tops_frob = 0.0

        self.file_save = file_save

        # Evaluate model
        self.__evaluate_synthetic_model()

        # Converse thetas to sparse
        thr = 3e-3
        self.local_model.doc_topic_distrib = \
            thetas2sparse(thr, self.local_model.doc_topic_distrib)
        self.gt_doc_topic_distrib = \
            thetas2sparse(thr, self.gt_doc_topic_distrib)

        # Save model
        save_model_as_npz(self.file_save, self)

    def __evaluate_synthetic_model(self):

        print(self.local_model.word_topic_distrib.shape)
        print(self.gt_word_topic_distrib.shape)
        all_words = ['wd'+str(word) for word in np.arange(self.vocab_size+1) if word > 0]
        self.local_model.word_topic_distrib = \
            convert_topic_word_to_init_size(self.vocab_size,
                                            self.local_model,
                                            "avitm",
                                            self.local_model.n_components,
                                            self.id2token,
                                            all_words)
        print(self.local_model.word_topic_distrib.shape)
        print(self.gt_word_topic_distrib.shape)
                  
        print('Tópicos (equivalentes) evaluados correctamente:', np.sum(np.max(np.sqrt(self.local_model.word_topic_distrib).dot(np.sqrt(self.gt_word_topic_distrib.T)), axis=0)))

        # Get thetas of the documents corresponding only to the node's corpus
        inic = (self.id-1)*len(self.gt_doc_topic_distrib)
        print(inic)
        end = (self.id)*len(self.gt_doc_topic_distrib)
        print(end)
        print(self.local_model.doc_topic_distrib.shape)
        thetas = self.local_model.doc_topic_distrib[inic:end, :]
        print(thetas)

        sim_mat_theoretical = np.sqrt(self.gt_doc_topic_distrib[0]).dot(np.sqrt(self.gt_doc_topic_distrib[0].T))
        sim_mat_actual = np.sqrt(thetas).dot(np.sqrt(thetas.T))
        print('Difference in evaluation of doc similarity:', np.sum(np.abs(sim_mat_theoretical - sim_mat_actual))/len(self.gt_doc_topic_distrib))