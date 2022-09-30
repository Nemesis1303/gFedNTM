import datetime
import multiprocessing as mp
import os
import warnings
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.special import softmax
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

# Local imports
from .decoding_network import DecoderNetwork
from ...utils.early_stopping.pytorchtools import EarlyStopping


class CTM(object):
    """Class to train the contextualized topic model. 
    This is the more general class, so in order to create one of the CTM models, the subclasses ZeroShotTM and CombinedTM should be used.
    """

    def __init__(self, logger, input_size, contextual_size,inference_type="combined", n_components=10, model_type='prodLDA', hidden_sizes=(100, 100), activation='softplus', dropout=0.2, learn_priors=True, batch_size=64, lr=2e-3, momentum=0.99, solver='adam', num_epochs=100, num_samples=10, reduce_on_plateau=False, topic_prior_mean=0.0, topic_prior_variance=None, num_data_loader_workers=mp.cpu_count(), label_size=0, loss_weights=None, verbose=True):

        """
        Sets the main attributes to create a specific CTM model.

        Parameters
        ----------
        logger: Logger  
            Object used to emit log messages
        input_size : int
            Dimension of the input
        contextual_size : int
            Dimension of the input that comes from the embeddings. 
            F.e. for BERT, it is 768
        inference_type: string (default='combined')
            Type of inference network to be used. It can be either 'combined' or 'contextual'
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
        num_samples: int (default=10)
            Number of times the theta needs to be sampled
        reduce_on_plateau: bool (default=False)
            If true, reduce learning rate by 10x on plateau of 10 epochs 
        topic_prior_mean: double (default=0.0)
            Mean parameter of the prior
        topic_prior_variance: double (default=None)
            Variance parameter of the prior
        num_data_loader_workers: int (default=0)
            Number of subprocesses to use for data loading
        label_size: int (default=0)
            Number of total labels
        loss_weights: dict
            It contains the name of the weight parameter (key) and the weight (value) for each loss.
        verbose: bool
            If True, additional logs are displayed
        """

        assert isinstance(input_size, int) and input_size > 0, \
            "input_size must by type int > 0."
        assert (isinstance(n_components, int) or isinstance(n_components, np.int64)) and n_components > 0, \
            "n_components must by type int > 0."
        assert model_type in ['LDA', 'prodLDA'], \
            "model must be 'LDA' or 'prodLDA'."
        assert isinstance(hidden_sizes, tuple), \
            "hidden_sizes must be type tuple."
        assert activation in ['softplus', 'relu', 'sigmoid', 'swish', 'tanh', 'leakyrelu',
                              'rrelu', 'elu', 'selu'], \
            "activation must be 'softplus', 'relu', 'sigmoid', 'swish', 'leakyrelu'," \
            " 'rrelu', 'elu', 'selu' or 'tanh'."
        assert dropout >= 0, "dropout must be >= 0."
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

        self.device = (
            torch.device("cuda")
            if torch.cuda.is_available()
            else torch.device("cpu")
        )

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
        self.num_samples = num_samples
        self.contextual_size = contextual_size
        self.momentum = momentum
        self.solver = solver
        self.num_epochs = num_epochs
        self.reduce_on_plateau = reduce_on_plateau
        self.num_data_loader_workers = num_data_loader_workers
        self.topic_prior_mean = topic_prior_mean
        self.topic_prior_variance = topic_prior_variance
        self.verbose = verbose

        # Performance attributes
        self.best_loss_train = float('inf')

        # Training attributes
        self.model_dir = None
        self.train_data = None
        self.nn_epoch = None

        # Larned topics
        self.best_components = None

        if loss_weights:
            self.weights = loss_weights
        else:
            self.weights = {"beta": 1}

        # Initialize inference avitm network
        self.model = DecoderNetwork(
            input_size, self.contextual_size, inference_type, n_components, model_type, hidden_sizes, activation,
            dropout, learn_priors, label_size=label_size)

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

        self.model = self.model.to(self.device)

    def _loss(self, inputs, word_dists, prior_mean, prior_variance,
              posterior_mean, posterior_variance, posterior_log_variance):

        """

        Defines the loss function utilized to optimize the parameter values in the AVITM model, and thus, in the CTM
        models.

        Parameters
        ----------
        inputs: torch.Tensor [self.batch_size, self.input_size]
            Data samples propagated through the AVITM network before the learnable parameters are updated
        word_dists: torch.nn.parameter.Parameter, [self.batch_size, self.input_size]
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
            Output of the encoder network that defines the covariance parameter (Σ1) of the logistic normal distribution used to approximate the Dirichlet prior ((p(θ|α)) according to the Laplace approximation
        
        Returns
        -------
        KL: torch.Tensor
            KL term
        RL: torch.Tensor
            Reconstruction term
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
        KL = 0.5 * (var_division + diff_term - self.n_components + logvar_det_division)

        #######################
        # Reconstruction term #
        #######################
        RL = -torch.sum(inputs * torch.log(word_dists + 1e-10), dim=1)

        return KL, RL

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
            X_bow = batch_samples['X_bow']
            X_bow = X_bow.reshape(X_bow.shape[0], -1)
            X_contextual = batch_samples['X_contextual']

            if "labels" in batch_samples.keys():
                labels = batch_samples["labels"]
                labels = labels.reshape(labels.shape[0], -1)
                labels = labels.to(self.device)
            else:
                labels = None

            if self.USE_CUDA:
                X_bow = X_bow.cuda()
                X_contextual = X_contextual.cuda()

            # forward pass
            self.model.zero_grad()
            prior_mean, prior_variance, \
            posterior_mean, posterior_variance, posterior_log_variance, \
            word_dists, estimated_labels = self.model(X_bow, X_contextual, labels)

            # backward pass
            kl_loss, rl_loss = self._loss(X_bow, word_dists, prior_mean, prior_variance, posterior_mean,
                                          posterior_variance, posterior_log_variance)

            loss = self.weights["beta"] * kl_loss + rl_loss
            loss = loss.sum()

            if labels is not None:
                target_labels = torch.argmax(labels, 1)

                label_loss = torch.nn.CrossEntropyLoss()(estimated_labels, target_labels)
                loss += label_loss

            loss.backward()
            self.optimizer.step()

            # compute train loss
            samples_processed += X_bow.size()[0]
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
            X_bow = batch_samples['X_bow']
            X_bow = X_bow.reshape(X_bow.shape[0], -1)
            X_contextual = batch_samples['X_contextual']

            if "labels" in batch_samples.keys():
                labels = batch_samples["labels"]
                labels = labels.to(self.device)
                labels = labels.reshape(labels.shape[0], -1)
            else:
                labels = None

            if self.USE_CUDA:
                X_bow = X_bow.cuda()
                X_contextual = X_contextual.cuda()

            # forward pass
            self.model.zero_grad()
            prior_mean, prior_variance, \
            posterior_mean, posterior_variance, posterior_log_variance, \
            word_dists, estimated_labels = self.model(X_bow, X_contextual, labels)

            kl_loss, rl_loss = self._loss(X_bow, word_dists, prior_mean, prior_variance,
                                          posterior_mean, posterior_variance, posterior_log_variance)

            loss = self.weights["beta"] * kl_loss + rl_loss
            loss = loss.sum()

            if labels is not None:
                target_labels = torch.argmax(labels, 1)
                label_loss = torch.nn.CrossEntropyLoss()(estimated_labels, target_labels)
                loss += label_loss

            # compute train loss
            samples_processed += X_bow.size()[0]
            val_loss += loss.item()

        val_loss /= samples_processed

        return samples_processed, val_loss

    def fit(self, train_dataset, validation_dataset=None, save_dir=None,
            patience=5, delta=0, n_samples=20):
        """
        Trains the CTM model.

        Parameters
        ----------
        train_dataset: CTMDataset
            Dataset used for training the CTM model.
        validation_dataset: CTMDataset
            Dataset used for validating the CTM model.
            If not None, the training stops if validation loss does not imporve after a given patience.
        save_dir: str (default=None)
            Directory to save checkpoint models to.
        patience: int (default=5)
            How long to wait after last time validation loss improved
        delta: int (default=0)
            Minimum change in the monitored quantity to qualify as an improvement
        n_samples: int (default=20)
            Number of samples of the document topic distribution 
        """

        # Print settings to output file
        if self.verbose:
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
            self.early_stopping = EarlyStopping(patience=patience, verbose=self.verbose, path=save_dir, delta=delta)

        train_loader = DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True,
                                  num_workers=self.num_data_loader_workers)

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
                    self.validation_data, batch_size=self.batch_size, shuffle=True,
                    num_workers=self.num_data_loader_workers)

                # train epoch
                s = datetime.datetime.now()
                val_samples_processed, val_loss = self._validate_epoch(validation_loader)
                e = datetime.datetime.now()

                # report
                if self.verbose:
                    self.logger.info("Epoch: [{}/{}]\tSamples: [{}/{}]\tValidation Loss: {}\tTime: {}".format(
                        epoch + 1, self.num_epochs, val_samples_processed,
                        len(self.validation_data) * self.num_epochs, val_loss, e - s))

                pbar.set_description(
                    "Epoch: [{}/{}]\t Seen Samples: [{}/{}]\tTrain Loss: {}\tValid Loss: {}\tTime: {}".format(
                        epoch + 1, self.num_epochs, samples_processed, len(self.train_data) * self.num_epochs,
                        train_loss, val_loss, e - s))

                if np.isnan(val_loss) or np.isnan(train_loss):
                    break
                else:
                    self.early_stopping(val_loss, self)
                    if self.early_stopping.early_stop:
                        self.logger.info("Early stopping")
                        break

            pbar.set_description(
                "Epoch: [{}/{}]\t Seen Samples: [{}/{}]\tTrain Loss: {}\tTime: {}".format(
                    epoch + 1, self.num_epochs, samples_processed, len(self.train_data) * self.num_epochs,
                    train_loss, e - s))

        pbar.close()
        self.training_doc_topic_distributions = self.get_doc_topic_distribution(train_dataset, n_samples)

    def get_predicted_topics(self, dataset, n_samples):
        """
        Returns the a list containing the predicted topic for each document (length: number of documents).
        
        Parameters
        ----------
        dataset: CTMDataset 
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
        dataset: CTMDataset
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
            num_workers=self.num_data_loader_workers)

        pbar = tqdm(n_samples, position=0, leave=True)

        final_thetas = []
        for sample_index in range(n_samples):

            with torch.no_grad():
                collect_theta = []

                for batch_samples in loader:
                    # batch_size x vocab_size
                    X_bow = batch_samples['X_bow']
                    X_bow = X_bow.reshape(X_bow.shape[0], -1)
                    X_contextual = batch_samples['X_contextual']

                    if "labels" in batch_samples.keys():
                        labels = batch_samples["labels"]
                        labels = labels.to(self.device)
                        labels = labels.reshape(labels.shape[0], -1)
                    else:
                        labels = None

                    if self.USE_CUDA:
                        X_bow = X_bow.cuda()
                        X_contextual = X_contextual.cuda()

                    # forward pass
                    self.model.zero_grad()
                    collect_theta.extend(self.model.get_theta(X_bow, X_contextual, labels).cpu().numpy().tolist())

                pbar.update(1)
                pbar.set_description("Sampling: [{}/{}]".format(sample_index + 1, n_samples))

                final_thetas.append(np.array(collect_theta))
        pbar.close()

        doct_topic_distrib = np.sum(final_thetas, axis=0) / n_samples
        return doct_topic_distrib

    def get_topic_word_matrix(self):
        """
        Gets the topic-word matrix associated with the trained model. If model_type is LDA, the matrix is normalized; otherwise, it is unnormalized.

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

    def get_word_distribution_by_topic_id(self, topic):
        """
        Gets the word probability distribution of a topic sorted by probability.

        Parameters
        ----------
        topic: int
            Identifier of the topic
       
        Returns
        -------
        t: List(tuple)
            List of tuples with the form (word, probability) sorted by the probability in descending order.
        """

        if topic >= self.n_components:
            raise Exception('Topic id must be lower than the number of topics')
        else:
            wd = self.get_topic_word_distribution()
            t = [(word, wd[topic][idx]) for idx, word in self.train_data.idx2token.items()]
            t = sorted(t, key=lambda x: -x[1])
        return t

    def get_top_documents_per_topic_id(self, unpreprocessed_corpus, document_topic_distributions, topic_id, k=5):
        """
        Gets the word probability distribution of a topic sorted by probability.

        # @ TODO: Complete docs
        Parameters
        ----------
        unpreprocessed_corpus: 

        document_topic_distributions:

        topic_id:

        k:
       
        Returns
        -------
        res: 

        """

        probability_list = document_topic_distributions.T[topic_id]
        ind = probability_list.argsort()[-k:][::-1]
        res = []
        for i in ind:
            res.append((unpreprocessed_corpus[i], document_topic_distributions[i][topic_id]))
        return res

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

    def get_most_likely_topic(self, doc_topic_distribution):
        """
        Gets the most likely topic for each document

        Parameters
        ----------
        doc_topic_distribution: ndarray
            Document-topic distribution

        Returns
        -------
        t: List[int]
           Most likely topics for each document
        """

        t = np.argmax(doc_topic_distribution, axis=0)
        return t

    def _format_file(self):
        model_dir = "contextualized_topic_model_nc_{}_tpm_{}_tpv_{}_hs_{}_ac_{}_do_{}_lr_{}_mo_{}_rp_{}". \
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

        self.model.load_state_dict(checkpoint['state_dict'])
    
    def get_ldavis_data_format(self, vocab, dataset, n_samples):
        """
        Returns the data that can be used in input to pyldavis to plot
        the topics
        """
        term_frequency = np.ravel(dataset.X_bow.sum(axis=0))
        self.logger.info(term_frequency.shape)
        doc_lengths = np.ravel(dataset.X_bow.sum(axis=1))
        self.logger.info(doc_lengths.shape)
        term_topic = self.get_topic_word_distribution()
        self.logger.info(term_topic.shape)
        doc_topic_distribution = self.get_doc_topic_distribution(
            dataset, n_samples=n_samples)
        self.logger.info(doc_topic_distribution.shape)
        self.logger.info(len(vocab))

        data = {'topic_term_dists': term_topic,
                'doc_topic_dists': doc_topic_distribution,
                'doc_lengths': doc_lengths,
                'vocab': vocab,
                'term_frequency': term_frequency}

        return data


"""
******************************************************************************
***                              ZeroShotTM                                ***
******************************************************************************
"""


class ZeroShotTM(CTM):
    """ZeroShotTM, as described in https://arxiv.org/pdf/2004.07737v1.pdf
    """

    def __init__(self, **kwargs):
        inference_type = "zeroshot"
        super().__init__(**kwargs, inference_type=inference_type)


"""
******************************************************************************
***                              CombinedTM                                ***
******************************************************************************
"""


class CombinedTM(CTM):
    """CombinedTM, as described in https://arxiv.org/pdf/2004.03974.pdf
    """

    def __init__(self, **kwargs):
        inference_type = "combined"
        super().__init__(**kwargs, inference_type=inference_type)
