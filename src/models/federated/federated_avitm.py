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
from src.models.base.pytorchavitm.datasets.bow_dataset import BOWDataset
from src.models.federated.federated_model import FederatedModel
from src.utils.auxiliary_functions import save_model_as_npz
from src.utils.utils_postprocessing import convert_topic_word_to_init_size


class FederatedAVITM(AVITM, FederatedModel):
    """Class for the Federated AVITM model.
    """

    def __init__(self,
                 tm_params: dict,
                 logger=None) -> None:

        FederatedModel.__init__(self, tm_params, logger)

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

    # ======================================================
    # Client-side training
    # ======================================================
    def train_mb_delta(self) -> dict:
        """Generates gradient update for a minibatch of samples.

        Returns
        -------
        params: dict
            Dictionary containing the parameters to be sent to the server.
        """

        # Get samples in minibatch
        self.X = self.current_batch_sample['X']

        if self.USE_CUDA:
            self.X = self.X.cuda()

        # Forward pass
        self.model.zero_grad()  # Update gradients to zero
        prior_mean, prior_var, posterior_mean,\
            posterior_var, posterior_log_var, \
            word_dists = self.model(self.X)

        # Backward pass: Compute gradients
        self.loss = self._loss(self.X, word_dists, prior_mean, prior_var,
                               posterior_mean, posterior_var, posterior_log_var)
        self.loss.backward()

        # Create gradient update to be sent to the server
        params = {
            "prior_mean": self.model.prior_mean,
            "prior_variance": self.model.prior_variance,
            "beta": self.model.beta,
            "current_mb": self.current_mb,
            "current_epoch": self.current_mb,
            "num_epochs": self.num_epochs
        }

        return params

    def deltaUpdateFit(self, modelStateDict, save_dir, n_samples=20) -> None:
        """Updates gradient with aggregated gradient from server and calculates loss.

        Parameters
        ----------
        modelStateDict: dict
            Dictionary containing the model parameters to be updated.
        """

        # Update local model's state dict
        localStateDict = self.model.state_dict()
        localStateDict["prior_mean"] = modelStateDict["prior_mean"]
        localStateDict["prior_variance"] = modelStateDict["prior_variance"]
        localStateDict["beta"] = modelStateDict["beta"]
        self.model.load_state_dict(localStateDict)

        # Calculate minibatch's train loss and samples processed
        self.samples_processed += self.X.size()[0]
        self.train_loss += self.loss.item()

        # Minitbatch ends
        self.current_mb += 1

        try:
            # Get next minibatch
            self.current_batch_sample = next(self.train_loader_iter)  # !!!!!!
        except StopIteration as ex:
            # If there is no next minibatch, the epoch has ended, so we report, reset the iterator and get the first one
            
            self.train_loss /= self.samples_processed
            
            print("Epoch: [{}/{}]\tSamples: [{}/{}]\tTrain Loss: {}\tTime: {}".format(
                self.current_epoch+1, self.num_epochs, self.samples_processed,
                len(self.train_data)*self.num_epochs, self.train_loss, datetime.datetime.now()))

            # Save best epoch results
            if self.current_epoch == 0:
                self.best_components = self.model.beta
            if self.train_loss < self.best_loss_train:
                self.best_components = self.model.beta
                self.best_loss_train = self.train_loss

            # Reset iterator and get first
            self.train_loader_iter = iter(self.train_loader)
            self.current_batch_sample = next(self.train_loader_iter)

            # Next epoch
            self.current_epoch += 1

        # Epoch end reached
        if self.current_epoch >= self.num_epochs:
            print("Epoch end reached")
            self.training_doc_topic_distributions = self.get_doc_topic_distribution(
                self.train_data, n_samples)
            self.get_results_model(save_dir)

        return

    # ======================================================
    # Server-side training
    # ======================================================
    def optimize_on_minibatch_from_server(self, updates) -> None:
        """Updates the gradients of the local model after aggregating the gradients from the clients.

        Parameters
        ----------
        updates: dict
            Dictionary with the gradients to be updated.
        """

        # Update model's parameters from the forward pass carried at client's side
        self.model.topic_word_matrix = self.model.beta
        self.best_components = self.model.beta
        
        # Upadate gradients after averaging from the clients'
        # Parameter0 = prior_mean
        # Parameter1 = prior_variance
        # Parameter2 = beta
        # self.model.prior_mean.grad = update
        # self.model.prior_variance = updates[0]
        # self.model.prior_variance = updates[1]
        # self.model.beta.grad = updates[2]
        self.model.prior_mean.grad = torch.Tensor(updates["prior_mean"])
        self.model.prior_variance.grad = torch.Tensor(
            updates["prior_variance"])
        self.model.beta.grad = torch.Tensor(updates["beta"])

        # Perform one step of the optimizer (SGD/Adam)
        self.optimizer.step()

        return

    # ======================================================
    # Evaluation
    # ======================================================
    def get_results_model(self, save_dir:str) -> None:
        """Gets the results of the model after training at the CLIENT side.
        
        Parameters
        ----------
        save_dir: str
            Directory where the model will be saved.
        """

        # Get topics
        self.topics = self.get_topics()
        
        # Get doc-topic distribution
        self.thetas = \
            np.asarray(self.get_doc_topic_distribution(self.train_data))
        self.thetas[self.thetas < 3e-3] = 0
        self.thetas = normalize(self.thetas, axis=1, norm='l1')

        # Get word-topic distribution
        self.betas = self.get_topic_word_distribution()

        #file_save = \
        #     "workspace/static/output_models/model_client_" + \
        #     str(self.fedTrManager.client.id) + ".npz"

        print("Saving model at: ", save_dir)
        save_model_as_npz(save_dir, self)
        
        return

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

    def get_topics_in_server(self, save_dir:str):
        """Gets the topics of the model after training at the SERVER side.
        The topics' chemical description cannot be inferred since the server does not have access to the training corpus. Inference is required in order to get the topic distribution.
        """
        
        # TODO: fix this
        # Get word-topic distribution
        self.betas = self.get_topic_word_distribution()
        print("Coge los topics in server")
        print(self.betas)
        
        print("Saving global model...")
        save_model_as_npz(save_dir, self)
        
