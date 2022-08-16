# -*- coding: utf-8 -*-
"""
******************************************************************************
***                             AVITM_model                                ***
******************************************************************************
"""
import os
from collections import defaultdict
import multiprocessing as mp
import requests
import numpy as np
import datetime
from sklearn.decomposition import non_negative_factorization
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from torch.nn import functional as F
from scipy.special import softmax
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split

from models.early_stopping.pytorchtools import EarlyStopping
from models.pytorchavitm.avitm.decoder_network import DecoderNetwork


class AVITM_model(object):

    """Class to train an AVITM model."""

    def __init__(self, input_size, n_components=10, model_type='prodLDA',
                 hidden_sizes=(100, 100), activation='softplus', dropout=0.2,
                 learn_priors=True, batch_size=64, lr=2e-3, momentum=0.99,
                 solver='adam', num_epochs=100, reduce_on_plateau=False, 
                 topic_prior_mean=0.0, topic_prior_variance=None,
                 num_samples=10, num_data_loader_workers=0, verbose=False):
        
        """
        Sets the main attributes to create a specific AVITM_model.

        Parameters
        ----------
        input_size : int
            Dimension of the input
        n_components : int (default=10)
            Number of topic components
        model_type : string (default='prodLDA')
            Type of the model that is going to be trained, 'prodLDA' or 'LDA'
        hidden_sizes : tuple, length = n_layers (default=(100,100))
            Size of the hidden layer
        activation : string (default='softplus')
            Activation function to be used, chosen from 'softplus', 'relu', 'sigmoid', 'leakyrelu', 'rrelu', 'elu', 'selu' or 'tanh'
        dropout : float (default=0.2)
            Percent of neurons to drop out.
        learn_priors : bool, (default=True)
            If true, priors are made learnable parameters
        lr: float (defualt=2e-3)
            Learning rate to be used for training
        momentum: folat (default=0.99)
            Momemtum to be used for training
        solver: string (default='adam')
            NN optimizer to be used, chosen from 'adagrad', 'adam', 'sgd', 'adadelta' or 'rmsprop' 
        num_epochs: int (default=100)
            Number of epochs to train for
        reduce_on_plateau: bool (default=False)
            If true, reduce learning rate by 10x on plateau of 10 epochs 
        topic_prior_mean: double (default=0.0)
            Mean parameter of the prior
        topic_prior_variance: double (default=None)
            Variance parameter of the prior
        num_samples: int (default=10)
            Number of sampling utilized during multiple sampling to redeuce variation
        num_data_loader_workers: int (default=0)
            Number of subprocesses to use for data loading.
        """

        assert isinstance(input_size, int) and input_size > 0,\
            "input_size must by type int > 0."
        assert isinstance(n_components, int) and input_size > 0,\
            "n_components must by type int > 0."
        assert model_type in ['LDA', 'prodLDA'],\
            "model must be 'LDA' or 'prodLDA'."
        assert isinstance(hidden_sizes, tuple), \
            "hidden_sizes must be type tuple."
        assert activation in ['softplus', 'relu', 'sigmoid', 'swish', 'tanh', 'leakyrelu',
                        'rrelu', 'elu', 'selu'], \
            "activation must be 'softplus', 'relu', 'sigmoid', 'swish', 'leakyrelu'," \
            " 'rrelu', 'elu', 'selu' or 'tanh'."
        assert dropout >= 0, "dropout must be >= 0."
        assert isinstance(learn_priors, bool), "learn_priors must be boolean."
        assert isinstance(batch_size, int) and batch_size > 0,\
            "batch_size must be int > 0."
        assert lr > 0, "lr must be > 0."
        assert isinstance(momentum, float) and momentum > 0 and momentum <= 1,\
            "momentum must be 0 < float <= 1."
        assert solver in ['adagrad', 'adam', 'sgd', 'adadelta', 'rmsprop'], \
            "solver must be 'adam', 'adadelta', 'sgd', 'rmsprop' or 'adagrad'"
        assert isinstance(reduce_on_plateau, bool),\
            "reduce_on_plateau must be type bool."
        assert isinstance(topic_prior_mean, float), \
            "topic_prior_mean must be type float"

        # General attributes
        self.input_size = input_size
        self.n_components = n_components
        self.model_type = model_type
        self.hidden_sizes = hidden_sizes
        self.activation = activation
        self.dropout = dropout
        self.learn_priors = learn_priors
        self.batch_size = batch_size
        self.lr = lr
        self.momentum = momentum
        self.solver = solver
        self.num_epochs = num_epochs
        self.reduce_on_plateau = reduce_on_plateau
        self.topic_prior_mean = topic_prior_mean
        self.topic_prior_variance = topic_prior_variance
        self.num_data_loader_workers = num_data_loader_workers
        self.num_samples = num_samples
        self.verbose = verbose

        # Performance attributes
        self.best_loss_train = float('inf')

        # Training atributes
        self.model_dir = None
        self.train_data = None
        self.nn_epoch = None

        # Learned topics
        self.best_components = None

        # Attribute to be used for saving the validation data
        self.validation_data = None

        # Initialize inference avitm network
        self.model = DecoderNetwork(
            input_size, n_components, model_type, hidden_sizes, activation,
            dropout, learn_priors)
        
        # Initialize Early Stopping method to stop the training if validation loss doesn't improve after a given patience.
        self.early_stopping = EarlyStopping(patience=5, verbose=False)

        # Initialize optimizer
        if self.solver == 'adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=lr, betas=(self.momentum, 0.99))
        elif self.solver == 'sgd':
            self.optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=self.momentum)
        elif self.solver == 'adagrad':
            self.optimizer = optim.Adagrad(self.model.parameters(), lr=lr)
        elif self.solver == 'adadelta':
            self.optimizer = optim.Adadelta(self.model.parameters(), lr=lr)
        elif self.solver == 'rmsprop':
            self.optimizer = optim.RMSprop(self.model.parameters(), lr=lr, momentum=self.momentum)

        # Initialize lr scheduler
        if self.reduce_on_plateau:
            self.scheduler = ReduceLROnPlateau(self.optimizer, patience=10)

        # Use cuda if available
        if torch.cuda.is_available():
            self.USE_CUDA = True
        else:
            self.USE_CUDA = False

        if self.USE_CUDA:
            self.model = self.model.cuda()
            

    def _loss(self, inputs, word_dists, prior_mean, prior_variance,
              posterior_mean, posterior_variance, posterior_log_variance):
        """
        Defines the loss function utlized to optimize the parameter values in the AVITM model.

        Parameters
        ----------
        inputs: torch.Tensor [self.batch_size, self.input_size]
            Data samples propagated through the AVITM network before the learnable parameters are updated
        word_dist: torch.nn.parameter.Parameter, [self.batch_size, self.input_size]
            Mixture of multinomials for each word
        prior_mean: torch.nn.parameter.Parameter, [self.n_components]
            Learnable parameter from the decoder network that defines the mean of the approximate variational posterior (q(θ)) defined as a logistic normal
        prior_variance: torch.Tensor [self.n_components]
             Learnable parameter from the decoder network that defines the covariance of the approximate variational posterior (q(θ)) defined as a logistic normal
        posterior_mean: torch.Tensor [self.batch_size, self.n_components]
            Output of the encoder network that defines the mean parameter (μ1) of the logistic normal distribution used to approximate the Dirichlet prior ((p(θ|α)) according to the Laplace approximation
        posterior_variance: torch.Tensor [self.batch_size, self.n_components]
            Exponential of the posterior_log_variance
        posterior_log_variance: torch.Tensor [self.batch_size, self.n_components]
            torch.Tensor [self.batch_size, self.n_components]
            Output of the encoder network that defines the covariance parameter (Σ1) of the logistic normal distribution used to approximate the Dirichlet prior ((p(θ|α))according to the Laplace approximation
        
        Returns:
        --------
        loss.sum(): torch.Tensor
            Accumulated loss for the samples in a batch 
        """

        ###########
        # KL term #
        ###########
        # Var division term
        var_division = torch.sum(posterior_variance / prior_variance, dim=1)
        
        # Diff means term
        diff_means = prior_mean - posterior_mean
        diff_term = torch.sum(
            (diff_means * diff_means) / prior_variance, dim=1)

        # Logvar det division term
        logvar_det_division = \
            prior_variance.log().sum() - posterior_log_variance.sum(dim=1)

        # Combine terms
        KL = 0.5 * (
            var_division + diff_term - self.n_components + logvar_det_division)

        #######################
        # Reconstruction term #
        #######################
        RL = -torch.sum(inputs * torch.log(word_dists + 1e-10), dim=1)

        loss = KL + RL
    
        return loss.sum()

    def _train_epoch(self, loader):
        """
        Trains an epoch.

        Parameters
        ----------
        loader: DataLoader
            Python iterable over the training dataset with which the epoch is going to be trained.

        Returns:
        --------
        samples_processed: int
            Number of processed samples
        train_loss: float
            Training loss
        topic_words: torch.nn.parameter.Parameter [self.n_components, self.input_size]
            Word-topic distribution for the current epoch
        topic_doc_list: List[torch.Tensor]
            List of topic-documents distributions for each batch in an epoch
        """

        self.model.train()

        train_loss = 0
        samples_processed = 0
        topic_doc_list = []

        for batch_samples in loader:
            # batch_size x vocab_size
            X = batch_samples['X']

            if self.USE_CUDA:
                X = X.cuda()

            # forward pass
            self.model.zero_grad()
            prior_mean, prior_var, posterior_mean, posterior_var, posterior_log_var, \
            word_dists, topic_words, topic_document = self.model(X)

            topic_doc_list.extend(topic_document)

            # backward pass
            loss = self._loss(X, word_dists, prior_mean, prior_var,
                              posterior_mean, posterior_var, posterior_log_var)
            loss.backward()
            self.optimizer.step()

            # compute train loss
            samples_processed += X.size()[0]
            train_loss += loss.item()

        train_loss /= samples_processed

        return samples_processed, train_loss, topic_words, topic_doc_list
    
    def _validate_epoch(self, loader):
        """
        Validates an epoch.

        Parameters
        ----------
        loader: DataLoader
            Python iterable over the validation dataset with which the epoch is going to be evaluated.
            
        Returns:
        --------
        samples_processed: int
            Number of processed samples
        val_loss: float
            Validation loss
        """
        self.model.eval()

        val_loss = 0
        samples_processed = 0

        for batch_samples in loader:
            # batch_size x vocab_size
            x = batch_samples['X']

            if self.USE_CUDA:
                x = x.cuda()

            # forward pass
            self.model.zero_grad()
            prior_mean, prior_var, posterior_mean, posterior_var, posterior_log_var, \
            word_dists, topic_word, topic_document = self.model(x)

            loss = self._loss(x, word_dists, prior_mean, prior_var,
                              posterior_mean, posterior_var, posterior_log_var)

            # compute validation loss
            samples_processed += x.size()[0]
            val_loss += loss.item()

        val_loss /= samples_processed

        return samples_processed, val_loss


    def fit(self, dataset ,save_dir=None):
        """
        Trains the AVITM model.

        Parameters
        ----------
        dataset: BOWDataset
            Dataset used for training and validationg the AVITM model.
        save_dir: str (default=None)
            Directory to save checkpoint models to.
        """

        if self.verbose:
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
                    self.n_components,  self.topic_prior_mean,
                    self.topic_prior_variance, self.model_type,
                    self.hidden_sizes, self.activation, self.dropout, self.learn_priors,
                    self.lr, self.momentum, self.reduce_on_plateau, save_dir))
        
        # Split data into training and validation
        self.train_data, self.validation_data = train_test_split(dataset, test_size=0.25, random_state=42) 

        self.model_dir = save_dir

        train_loader = DataLoader(
            self.train_data, batch_size=self.batch_size, shuffle=True,
            num_workers=mp.cpu_count())

        # init training variables
        train_loss = 0
        samples_processed = 0

        # train loop
        for epoch in range(self.num_epochs):
            self.nn_epoch = epoch
            # train epoch
            s = datetime.datetime.now()
            sp, train_loss, topic_words, topic_document = self._train_epoch(train_loader)
            samples_processed += sp
            e = datetime.datetime.now()

            # report
            print("Epoch: [{}/{}]\tSamples: [{}/{}]\tTrain Loss: {}\tTime: {}".format(
                epoch+1, self.num_epochs, samples_processed,
                len(self.train_data)*self.num_epochs, train_loss, e - s))
            
            self.best_components = self.model.beta
            self.final_topic_word = topic_words
            self.final_topic_document = topic_document
            self.best_loss_train = train_loss

            if self.validation_data is not None:

                validation_loader = DataLoader(
                    self.validation_data, batch_size=self.batch_size, shuffle=True,
                    num_workers=mp.cpu_count())

                # train epoch
                s = datetime.datetime.now()
                val_samples_processed, val_loss = self._validate_epoch(validation_loader)
                e = datetime.datetime.now()

                # report
                print("Epoch: [{}/{}]\tSamples: [{}/{}]\tValidation Loss: {}\tTime: {}".format(
                    epoch + 1, self.num_epochs, val_samples_processed,
                    len(self.validation_data) * self.num_epochs, val_loss, e - s))
                
                if np.isnan(val_loss) or np.isnan(train_loss):
                    break
                else:
                    self.early_stopping(val_loss, self.model)
                    if self.early_stopping.early_stop:
                        print("Early stopping")
                        if save_dir is not None:
                            self.save(save_dir)
                        break


    def predict(self, dataset, k=10):
        """
        Generates predictions for a specific dataset. 

        Parameters
        ----------
        dataset: BOWDataset
            Dataset used for evaluation of the AVITM model
            
        Returns:
        --------
        preds: int

        test_doc_topic: float
            Document-topic distribution for the given dataset
        """
        self.model.eval()

        loader = DataLoader(
            dataset, batch_size=self.batch_size, shuffle=False,
            num_workers=mp.cpu_count())

        with torch.no_grad():
            for batch_samples in loader:
                # batch_size x vocab_size
                x = batch_samples['X']
                x = x.reshape(x.shape[0], -1)
                if self.USE_CUDA:
                    x = x.cuda()
                # forward pass
                self.model.zero_grad()
                _, _, _, _, _, _, _, topic_document = self.model(x)

                _, indices = torch.sort(topic_document, dim=1)
                preds += [indices[:, :k]]
            
            preds = torch.cat(preds, dim=0)

        test_doc_topic = np.asarray(self.get_thetas(dataset)).T

        return preds, test_doc_topic


    def get_topics(self, k=10):
        """
        Retrieves the k most significant words belonging to each of the trained topics.

        Parameters
        ----------
        k: int (default=10)
            Number of words to return per topic.

        Returns:
        --------
        topics_list: List[List[str]]
            List of the model's topics, each of them described as a list with the k most significant words belonging to it
        """
       
        assert k <= self.input_size, "k must be <= input size."
        component_dists = self.best_components
        topics = defaultdict(list)
        topics_list = []
        if self.n_components is not None:
            for i in range(self.n_components):
                _, idxs = torch.topk(component_dists[i], k)
                component_words = [self.train_data.idx2token[idx]
                                   for idx in idxs.cpu().numpy()]
                topics[i] = component_words
                topics_list.append(component_words)

        return topics_list

    def get_topic_word_distribution(self):
        """
        Gets the topic-word distribution associated with the trained model

        Returns:
        --------
        topic_word_distrib: numpy.ndarray
            A CxV matrix where C is the number of components and V is the vocabulary length
        """  
        wd = self.final_topic_word.cpu().detach().numpy()
        topic_word_distrib = softmax(wd, axis=1)
        return topic_word_distrib


    def get_document_topic_distribution(self):
        """
        Gets the document-topic distribution associated with the trained model

        Returns:
        --------
        top_doc_arr: numpy.ndarray
            A CxD matrix where C is the number of components and D is number of documents in the dataset
        """  
        top_doc = self.final_topic_document
        top_doc_arr = np.array([i.cpu().detach().numpy() for i in top_doc])
        return top_doc_arr


    def get_thetas(self, dataset, n_samples=20):
        """
        Given a trained AVITM model, it gets the associated document-topic distribution for a specific dataset.

        Parameters
        ----------
        dataset: BOWDataset
            Dataset whose document-topic distribution is being calculated
        n_samples: int (default=10)
            Number of sampling utilized during multiple sampling to redeuce variation

        Returns:
        --------
        top_doc_arr: numpy.ndarray
            A CxD matrix where C is the number of components and D is number of documents in the dataset given as argument
        """
 
        self.model.eval()

        loader = DataLoader(
                dataset, batch_size=self.batch_size, shuffle=False,
                num_workers=mp.cpu_count())

        pbar = tqdm(n_samples, position=0, leave=True)

        final_thetas = []
        for sample_index in range(self.num_samples):

            with torch.no_grad():
                collect_theta = []

                for batch_samples in loader:
                    # batch_size x vocab_size
                    x = batch_samples['X']
                    x = x.reshape(x.shape[0], -1)
                    if self.USE_CUDA:
                        x = x.cuda()
                    # forward pass
                    self.model.zero_grad()
                    collect_theta.extend(self.model.get_theta(x).cpu().numpy().tolist())

                pbar.update(1)
                pbar.set_description("Sampling: [{}/{}]".format(sample_index + 1, n_samples))

                final_thetas.append(np.array(collect_theta))

        pbar.close()

        return np.sum(final_thetas, axis=0) / n_samples


    def _format_file(self):
        """
        Formats the file in which the trained model will be saved if specified

        Returns:
        --------
        model_dir: os.PathLike object
            Directory in which the trained model will be saved
        """
        model_dir = "AVITM_nc_{}_tpm_{}_tpv_{}_hs_{}_ac_{}_do_{}_lr_{}_mo_{}_rp_{}".\
            format(self.n_components, 0.0, 1 - (1./self.n_components),
                   self.model_type, self.hidden_sizes, self.activation,
                   self.dropout, self.lr, self.momentum,
                   self.reduce_on_plateau)
        return model_dir


    def save(self, models_dir=None):
        """
        Saves the trained model as Pytorch object

        Parameters
        ----------
        model_dir: os.PathLike object
            Path to directory for saving NN models.
        """
        if (self.model is not None) and (models_dir is not None):

            model_dir = self._format_file()
            if not os.path.isdir(os.path.join(models_dir, model_dir)):
                os.makedirs(os.path.join(models_dir, model_dir))

            filename = "epoch_{}".format(self.nn_epoch) + '.pth'
            fileloc = os.path.join(models_dir, model_dir, filename)
            with open(fileloc, 'wb') as file:
                torch.save({'state_dict': self.model.state_dict(),
                            'dcue_dict': self.__dict__}, file)


    def load(self, model_dir, epoch):
        """
        Loads a previously trained model.

        Parameters
        ----------
        model_dir: os.PathLike object
            Path to directory for saving NN models.
        epoch: int
            Epoch of model to load.
        """
       
        epoch_file = "epoch_"+str(epoch)+".pth"
        model_file = os.path.join(model_dir, epoch_file)
        with open(model_file, 'rb') as model_dict:
            checkpoint = torch.load(model_dict)

        for (k, v) in checkpoint['dcue_dict'].items():
            setattr(self, k, v)

        self._init_nn()
        self.model.load_state_dict(checkpoint['state_dict'])
