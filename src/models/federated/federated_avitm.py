"""
Created on Feb 1, 2022

@author: L. Calvo-Bartolomé (lcalvo@pa.uc3m.es)
"""
import datetime

import numpy as np
import torch
from sklearn.preprocessing import normalize
from torch.utils.data import DataLoader

from src.models.base.pytorchavitm.avitm_network.avitm import AVITM
from src.models.federated.federated_model import FederatedModel
from src.utils.auxiliary_functions import save_model_as_npz
from src.utils.utils_postprocessing import convert_topic_word_to_init_size


class FederatedAVITM(AVITM, FederatedModel):
    def __init__(self, tm_params, client, logger):

        FederatedModel.__init__(self, tm_params, client, logger)

        AVITM.__init__(
            self,
            logger=self.logger,
            input_size=tm_params["input_size"],
            n_components=tm_params["n_components"],
            model_type=tm_params["model_type"],
            hidden_sizes=tm_params["hidden_sizes"],
            activation=tm_params["activation"],
            dropout=tm_params["dropout"],
            learn_priors=tm_params["learn_priors"],
            batch_size=tm_params["batch_size"],
            lr=tm_params["lr"],
            momentum=tm_params["momentum"],
            solver=tm_params["solver"],
            num_epochs=tm_params["num_epochs"],
            reduce_on_plateau=tm_params["reduce_on_plateau"],
            topic_prior_mean=tm_params["topic_prior_mean"],
            topic_prior_variance=tm_params["topic_prior_variance"],
            num_samples=tm_params["num_samples"],
            num_data_loader_workers=tm_params["num_data_loader_workers"],
            verbose=tm_params["verbose"])

        # Current epoch for tracking federated model
        self.current_mb = -1

        # Training set info parameters
        self.train_data = None
        self.id2token = self.tm_params["id2token"]

        # Post-training parameters
        self.topics = None
        self.thetas = None
        self.betas = None

    def _train_minibatch(self, X, train_loss, samples_processed):

        if self.USE_CUDA:
            X = X.cuda()

        # Forward pass
        self.model.zero_grad()  # Update gradients to zero
        prior_mean, prior_var, posterior_mean, posterior_var, posterior_log_var, word_dists = self.model(
            X)

        # Backward pass: Compute gradients
        loss = self._loss(X, word_dists, prior_mean, prior_var,
                          posterior_mean, posterior_var, posterior_log_var)
        loss.backward()

        return loss, train_loss, samples_processed

    def optimize_on_minibatch_from_server(self, updates):

        # Upadate gradients
        self.model.prior_mean.grad = torch.Tensor(updates["prior_mean"])
        self.model.prior_variance.grad = torch.Tensor(
            updates["prior_variance"])
        self.model.beta.grad = torch.Tensor(updates["beta"])

        # Perform one step of the optimizer (SGD/Adam)
        self.optimizer.step()

        return

    def _optimize_on_minibatch(self, X, loss, train_loss, samples_processed):
        # Update gradients
        # Parameter0 = prior_mean
        # Parameter1 = prior_variance
        # Parameter2 = beta
        #self.model.prior_mean.grad = update
        #self.model.prior_variance = updates[0]
        #self.model.prior_variance = updates[1]
        #self.model.beta.grad = updates[2]

        # Perform one step of the optimizer (SGD/Adam)
        # self.optimizer.step()

        # Compute train loss
        samples_processed += X.size()[0]
        train_loss += loss.item()

        return train_loss, samples_processed

    def _train_epoch(self, loader):
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

        self.model.train()

        train_loss = 0
        samples_processed = 0

        # Counter for the current minibatch
        self.current_mb = 0

        # Training epoch starts
        for batch_samples in loader:

            # Get samples in minibatch
            X = batch_samples['X']

            # Get gradients minibatch
            loss, train_loss, samples_processed = \
                self._train_minibatch(
                    X, train_loss, samples_processed)

            # Send minibatch' gradient to the server (gradient already converted to np)
            params = {
                "prior_mean": self.model.prior_mean,
                "prior_variance": self.model.prior_variance,
                "beta": self.model.beta,
                "current_mb": self.current_mb,
                "current_epoch": self.current_mb,
                "num_epochs": self.num_epochs
            }

            self.fedTrManager.send_gradient_minibatch(params)

            self.logger.info(
                'Client %s sent gradient %s/%s and is waiting for updates.',
                str(self.fedTrManager.client.id), str(self.current_mb),
                str(self.current_epoch))

            # Update model with server's weigths
            modelStateDict, optStateDict = self.fedTrManager.get_update_minibatch()

            localStateDict = self.model.state_dict()
            localStateDict["prior_mean"] = modelStateDict["prior_mean"]
            localStateDict["prior_variance"] = modelStateDict["prior_variance"]
            localStateDict["beta"] = modelStateDict["beta"]

            #modelStateDict["topic_word_matrix"] = self.model.topic_word_matrix
            #self.model.prior_mean = modelStateDict["prior_mean"]
            #self.model.prior_variance = modelStateDict["prior_variance"]
            #self.model.beta = nn.Parameter(modelStateDict["beta"])
            self.model.load_state_dict(localStateDict)
            # self.optimizer.load_state_dict(optStateDict)

            # Calculate minibatch's train loss and samples processed
            train_loss, samples_processed = \
                self._optimize_on_minibatch(
                    X, loss, train_loss, samples_processed)

            self.current_mb += 1  # One minibatch ends
        # Training epoch ends

        # Calculate epoch's train loss and samples processed
        train_loss /= samples_processed

        return samples_processed, train_loss

    def fit(self, train_data, save_dir=None):
        """
        Trains a federated AVITM model. 

        Parameters
        ----------
        save_dir: pathlib.Path, optional
            Directory to save checkpoint models to, defaults to None.
        """

        self.model_dir = save_dir
        self.train_data = train_data
        # TODO: Include validation data
        self.current_minibatch = 0
        self.current_epoch = 0

        train_loader = DataLoader(
            self.train_data, batch_size=self.batch_size,
            shuffle=True, num_workers=0)

        # Initialize training variables
        train_loss = 0
        samples_processed = 0

        # Training of the local model
        for epoch in range(self.num_epochs):
            self.current_epoch = epoch

            # Train epoch
            s = datetime.datetime.now()
            sp, train_loss = \
                self._train_epoch(train_loader)
            samples_processed += sp
            e = datetime.datetime.now()

            # report
            print("Epoch: [{}/{}]\tSamples: [{}/{}]\tTrain Loss: {}\tTime: {}".format(
                epoch+1, self.num_epochs, samples_processed,
                len(self.train_data)*self.num_epochs, train_loss, e - s))

            # save best
            if epoch == 0:
                print("ENTRA EN EPOCH 0")
                self.best_components = self.model.beta
            if train_loss < self.best_loss_train:
                self.best_components = self.model.beta
                self.best_loss_train = train_loss

                if save_dir is not None:
                    self.save(save_dir)

    def get_results_model(self):

        # Get topics
        self.topics = self.get_topics()

        # Get doc-topic distribution
        self.thetas = \
            np.asarray(self.get_doc_topic_distribution(self.train_data))
        self.thetas[self.thetas < 3e-3] = 0
        self.thetas = normalize(self.thetas, axis=1, norm='l1')

        # Get word-topic distribution
        self.betas = self.get_topic_word_distribution()

        file_save = \
            "workspace/data/output_models/model_client_" + \
            str(self.fedTrManager.client.id) + ".npz"

        save_model_as_npz(file_save, self)

    def evaluate_synthetic_model(self, vocab_size, gt_thetas, gt_betas):

        all_words = \
            ['wd'+str(word) for word in np.arange(vocab_size+1)
             if word > 0]

        self.betas = convert_topic_word_to_init_size(
            vocab_size=vocab_size,
            model=self,
            model_type="avitm",
            ntopics=self.n_components,
            id2token=self.tm_params["id2token"],
            all_words=all_words)
        print('Tópicos (equivalentes) evaluados correctamente:', np.sum(
            np.max(np.sqrt(self.betas).dot(np.sqrt(gt_betas.T)), axis=0)))

        sim_mat_theoretical = \
            np.sqrt(gt_thetas[0]).dot(np.sqrt(gt_thetas[0].T))
        sim_mat_actual = np.sqrt(self.thetas).dot(np.sqrt(self.thetas.T))
        print('Difference in evaluation of doc similarity:', np.sum(
            np.abs(sim_mat_theoretical - sim_mat_actual))/len(gt_thetas))

    def get_topics_in_server(self):
        # TODO: fix this
        #self.topics = self.get_topics()
        #print(self.topics)
        pass
