# -*- coding: utf-8 -*-
import datetime
import multiprocessing as mp
import os
from collections import defaultdict

import numpy as np
import torch
# Local imports
from src.models.base.utils.early_stopping.pytorchtools import EarlyStopping
from scipy.special import softmax
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from .decoder_network import DecoderNetwork


class AVITM(object):
    """Class to train an AVITM model."""

    def __init__(self, logger, input_size, n_components=10,
                 model_type='prodLDA', hidden_sizes=(100, 100), activation='softplus', dropout=0.2,
                 learn_priors=True, batch_size=64, lr=2e-3,
                 momentum=0.99, solver='adam', num_epochs=100, reduce_on_plateau=False, topic_prior_mean=0.0, topic_prior_variance=None, num_samples=10, num_data_loader_workers=0, verbose=True):
        """
        Sets the main attributes to create a specific AVITM.

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
            Activation function to be used, chosen from 'softplus', 'relu', 'sigmoid', 'leakyrelu', 'rrelu', 'elu',
            'selu' or 'tanh'
        dropout : float (default=0.2)
            Percent of neurons to drop out.
        learn_priors : bool, (default=True)
            If true, priors are made learnable parameters
        batch_size : int (default=64)
            Size of the batch to use for training
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
            Number of times the theta needs to be sampled
        num_data_loader_workers: int (default=0)
            Number of subprocesses to use for data loading
        verbose: bool
            If True, additional logs are displayed
        """

        assert isinstance(input_size, int) and input_size > 0, \
            "input_size must by type int > 0."
        assert isinstance(n_components, int) and input_size > 0, \
            "n_components must by type int > 0."
        assert model_type.lower() in ['lda', 'prodlda'], \
            "model must be 'LDA' or 'prodLDA'."
        assert isinstance(hidden_sizes, tuple), \
            "hidden_sizes must be type tuple."
        assert activation in ['softplus', 'relu', 'sigmoid', 'swish', 'tanh', 'leakyrelu', 'rrelu', 'elu', 'selu'], \
            "activation must be 'softplus', 'relu', 'sigmoid', 'swish', 'leakyrelu'," \
            " 'rrelu', 'elu', 'selu' or 'tanh'."
        assert dropout >= 0, "dropout must be >= 0."
        assert isinstance(learn_priors, bool), "learn_priors must be boolean."
        assert isinstance(batch_size, int) and batch_size > 0, \
            "batch_size must be int > 0."
        assert lr > 0, "lr must be > 0."
        assert isinstance(momentum, float) and momentum > 0 and momentum <= 1, \
            "momentum must be 0 < float <= 1."
        assert solver in ['adagrad', 'adam', 'sgd', 'adadelta', 'rmsprop'], \
            "solver must be 'adam', 'adadelta', 'sgd', 'rmsprop' or 'adagrad'"
        assert isinstance(reduce_on_plateau, bool), \
            "reduce_on_plateau must be type bool."
        assert isinstance(topic_prior_mean, float), \
            "topic_prior_mean must be type float"

        # General attributes
        self.logger = logger
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

        # Initialize Early Stopping method to stop the training if validation loss doesn't improve after a given
        # patience.
        self.early_stopping = EarlyStopping(patience=5, verbose=False)

        # Initialize optimizer
        if self.solver == 'adam':
            self.optimizer = optim.Adam(
                self.model.parameters(), lr=lr, betas=(self.momentum, 0.99))
        elif self.solver == 'sgd':
            self.optimizer = optim.SGD(
                self.model.parameters(), lr=lr, momentum=self.momentum)
        elif self.solver == 'adagrad':
            self.optimizer = optim.Adagrad(self.model.parameters(), lr=lr)
        elif self.solver == 'adadelta':
            self.optimizer = optim.Adadelta(self.model.parameters(), lr=lr)
        elif self.solver == 'rmsprop':
            self.optimizer = optim.RMSprop(
                self.model.parameters(), lr=lr, momentum=self.momentum)

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
        Defines the loss function utilized to optimize the parameter values in the AVITM model.

        Parameters
        ----------
        inputs: torch.Tensor
            Data samples propagated through the AVITM network before the learnable parameters are updated
            Dim: [self.batch_size, self.input_size]
        word_dists: torch.nn.parameter.Parameter
            Mixture of multinomials for each word
            Dim: [self.batch_size, self.input_size]
        prior_mean: torch.nn.parameter.Parameter
            Learnable parameter from the decoder network that defines the mean of the approximate variational posterior (q(θ)) defined as a logistic normal
            Dim: [self.n_components]
        prior_variance: torch.Tensor
            Learnable parameter from the decoder network that defines the covariance of the approximate variational posterior (q(θ)) defined as a logistic normal
            Dim: [self.n_components]
        posterior_mean: torch.Tensor
            Output of the encoder network that defines the mean parameter (μ1) of the logistic normal distribution used to approximate the Dirichlet prior ((p(θ|α)) according to the Laplace approximation
            Dim: [self.batch_size, self.n_components]
        posterior_variance: torch.Tensor 
            Exponential of the posterior_log_variance 
            Dim: [self.batch_size, self.n_components]
        posterior_log_variance: torch.Tensor 
            Output of the encoder network that defines the covariance parameter (Σ1) of the logistic normal distribution used to approximate the Dirichlet prior ((p(θ|α)) according to the Laplace approximation
            Dim: [self.batch_size, self.n_components]

        Returns
        -------
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

        Returns
        -------
        samples_processed: int
            Number of processed samples
        train_loss: float
            Training loss
        """

        self.model.train()

        train_loss = 0
        samples_processed = 0

        for batch_samples in loader:
            # batch_size x vocab_size
            X = batch_samples['X']

            if self.USE_CUDA:
                X = X.cuda()

            # forward pass
            self.model.zero_grad()
            prior_mean, prior_var, posterior_mean, posterior_var, posterior_log_var, word_dists = self.model(
                X)

            # backward pass
            loss = self._loss(X, word_dists, prior_mean, prior_var,
                              posterior_mean, posterior_var, posterior_log_var)
            loss.backward()
            self.optimizer.step()

            # compute train loss
            samples_processed += X.size()[0]
            train_loss += loss.item()

        train_loss /= samples_processed

        return samples_processed, train_loss

    def _validate_epoch(self, loader):
        """
        Validates an epoch.

        Parameters
        ----------
        loader: DataLoader
            Python iterable over the validation dataset with which the epoch is going to be evaluated.

        Returns
        -------
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
            prior_mean, prior_var, posterior_mean, posterior_var, posterior_log_var, word_dists = self.model(
                x)

            loss = self._loss(x, word_dists, prior_mean, prior_var,
                              posterior_mean, posterior_var, posterior_log_var)

            # compute validation loss
            samples_processed += x.size()[0]
            val_loss += loss.item()

        val_loss /= samples_processed

        return samples_processed, val_loss

    def fit(self, train_dataset, validation_dataset=None, save_dir=None, patience=5, delta=0, n_samples=20):
        """
        Trains the AVITM model.

        Parameters
        ----------
        train_dataset: BOWDataset
            Dataset used for training the AVITM model.
        validation_dataset: BOWDataset
            Dataset used for validating the AVITM model.
        save_dir: str (default=None)
            Directory to save checkpoint models to.
        patience: int (default=5)
            How long to wait after last time validation loss improved
        delta: int (default=0)
            Minimum change in the monitored quantity to qualify as an improvement
        n_samples: int (default=20)
            Number of samples of the document topic distribution 
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
                self.n_components, self.topic_prior_mean,
                self.topic_prior_variance, self.model_type,
                self.hidden_sizes, self.activation, self.dropout, self.learn_priors,
                self.lr, self.momentum, self.reduce_on_plateau, save_dir))

        self.model_dir = save_dir
        self.train_data = train_dataset
        self.validation_data = validation_dataset

        if self.validation_data is not None:
            self.early_stopping = EarlyStopping(
                patience=patience, verbose=self.verbose, path=save_dir, delta=delta)

        train_loader = DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=mp.cpu_count())

        # init training variables
        train_loss = 0
        samples_processed = 0

        # train loop
        pbar = tqdm(self.num_epochs, position=0, leave=True)
        for epoch in range(self.num_epochs):
            self.nn_epoch = epoch
            # train epoch
            s = datetime.datetime.now()
            sp, train_loss = self._train_epoch(train_loader)
            samples_processed += sp
            e = datetime.datetime.now()
            pbar.update(1)
            
            self.best_components = self.model.beta

            if self.validation_data is not None:

                validation_loader = DataLoader(
                    self.validation_data,
                    batch_size=self.batch_size,
                    shuffle=True,
                    num_workers=mp.cpu_count())

                # train epoch
                s = datetime.datetime.now()
                val_samples_processed, val_loss = \
                    self._validate_epoch(validation_loader)
                e = datetime.datetime.now()

                # report
                if self.verbose:
                    self.logger.info("Epoch: [{}/{}]\tSamples: [{}/{}]\tValidation Loss: {}\tTime: {}".format(
                        epoch + 1, self.num_epochs, val_samples_processed,
                        len(self.validation_data) * self.num_epochs, val_loss, e - s))

                pbar.set_description(
                    "Epoch: [{}/{}]\t Seen Samples: [{}/{}]\tTrain Loss: {}\tValid Loss: {}\tTime: {}".format(
                        epoch +
                        1, self.num_epochs, samples_processed, len(
                            self.train_data) * self.num_epochs,
                        train_loss, val_loss, e - s))
                
                if np.isnan(val_loss) or np.isnan(train_loss):
                    break
                else:
                    self.early_stopping(val_loss, self)
                    if self.early_stopping.early_stop:
                        self.logger.info("Early stopping")
                        break
                    
            else:
                # Save last epoch
                if save_dir is not None:
                    self.save(save_dir)

            pbar.set_description(
                "Epoch: [{}/{}]\t Seen Samples: [{}/{}]\tTrain Loss: {}\tTime: {}".format(
                    epoch +
                    1, self.num_epochs, samples_processed, len(
                        self.train_data) * self.num_epochs,
                    train_loss, e - s))

        pbar.close()
        self.training_doc_topic_distributions = self.get_doc_topic_distribution(
            train_dataset, n_samples)

    def get_predicted_topics(self, dataset, n_samples):
        """
        Returns the a list containing the predicted topic for each document (length: number of documents).

        Parameters
        ----------
        dataset: BOWDataset 
            Dataset to infer topics
        n_samples: int 
            Number of sampling of theta

        Returns
        -------
        predicted_topics: List
            the predicted topics
        """

        predicted_topics = []
        thetas = self.get_doc_topic_distribution(dataset, n_samples)

        for idd in range(len(dataset)):
            predicted_topic = np.argmax(thetas[idd] / np.sum(thetas[idd]))
            predicted_topics.append(predicted_topic)
        return predicted_topics

    def get_doc_topic_distribution(self, dataset, n_samples=20):
        """
        Gets the document-topic distribution for a dataset of topics. Includes multiple sampling to reduce variation via

        Parameters
        ----------
        dataset: BOWDataset
            Dataset whose document-topic distribution is being calculated
        n_samples: int (default=20)
           Number of sample to collect to estimate the final distribution (the more the better).

        Returns
        -------
        doct_topic_distrib: numpy.ndarray
            A CxD matrix where C is the number of components and D is number of documents in the dataset given as argument
        """

        self.model.eval()

        loader = DataLoader(
            dataset, batch_size=self.batch_size, shuffle=False,
            num_workers=mp.cpu_count())

        pbar = tqdm(n_samples, position=0, leave=True)

        final_thetas = []
        for sample_index in range(n_samples):

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
                    collect_theta.extend(
                        self.model.get_theta(x).cpu().numpy().tolist())

                pbar.update(1)
                pbar.set_description(
                    "Sampling: [{}/{}]".format(sample_index + 1, n_samples))

                final_thetas.append(np.array(collect_theta))
        pbar.close()

        doct_topic_distrib = np.sum(final_thetas, axis=0) / n_samples
        print(doct_topic_distrib)
        return doct_topic_distrib

    def get_topic_word_matrix(self):
        """
        Gets the topic-word matrix associated with the trained model. If model_type is LDA, the matrix is normalized;
        otherwise, it is unnormalized.

        Returns
        -------
        topic_word_mat: numpy.ndarray
            A CxV matrix where C is the number of components and V is the vocabulary length
        """

        topic_word_mat = self.model.topic_word_matrix.cpu().detach().numpy()
        return topic_word_mat

    def get_topic_word_distribution(self):
        """
        Gets the topic-word distriubtion associated with the trained model.

        Returns
        -------
        topic_word_distrib: numpy.ndarray
            A CxV matrix where C is the number of components and V is the vocabulary length
        """

        mat = self.get_topic_word_matrix()
        topic_word_distrib = softmax(mat, axis=1)
        return topic_word_distrib

    def get_topics(self, k=10):
        """
        Retrieves the k most significant words belonging to each of the trained topics.

        Parameters
        ----------
        k: int (default=10)
            Number of words to return per topic.

        Returns
        -------
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

    def _format_file(self):
        """
        Formats the file in which the trained model will be saved if specified

        Returns
        -------
        model_dir: os.PathLike object
            Directory in which the trained model will be saved
        """
        model_dir = "AVITM_nc_{}_tpm_{}_tpv_{}_hs_{}_ac_{}_do_{}_lr_{}_mo_{}_rp_{}". \
            format(self.n_components, 0.0, 1 - (1. / self.n_components),
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

        epoch_file = "epoch_" + str(epoch) + ".pth"
        model_file = os.path.join(model_dir, epoch_file)
        with open(model_file, 'rb') as model_dict:
            checkpoint = torch.load(model_dict)

        for (k, v) in checkpoint['dcue_dict'].items():
            setattr(self, k, v)

        self._init_nn()
        self.model.load_state_dict(checkpoint['state_dict'])
