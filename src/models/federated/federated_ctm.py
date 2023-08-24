"""
Created on Feb 1, 2022
Last updated on Aug 24, 2023

.. codeauthor:: L. Calvo-Bartolomé (lcalvo@pa.uc3m.es)
"""
import datetime
import torch
from src.models.base.contextualized_topic_models.ctm_network.ctm import CTM
from src.models.federated.federated_model import FederatedModel


class FederatedCTM(CTM, FederatedModel):
    """Class for the Federated CTM model.
    """
    def __init__(
            self,
            tm_params: dict,
            logger=None
        ) -> None:
        
        FederatedModel.__init__(self, tm_params, logger)

        CTM.__init__(
            self,
            logger=self.logger,
            input_size=tm_params["input_size"],
            contextual_size=tm_params["contextual_size"],
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
            num_samples=tm_params["num_samples"],
            reduce_on_plateau=tm_params["reduce_on_plateau"],
            topic_prior_mean=tm_params["topic_prior_mean"],
            topic_prior_variance=tm_params["topic_prior_variance"],
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
        self.X_bow = self.current_batch_sample['X_bow']
        self.X_bow = self.X_bow.reshape(self.X_bow.shape[0], -1)
        self.X_contextual = self.current_batch_sample['X_contextual']
        
        if "labels" in self.current_batch_sample.keys():
            self.labels = self.current_batch_sample['labels']
            self.labels = self.labels.reshape(self.labels.shape[0], -1)
            self.labels = self.labels.to(self.device)
        else:
            self.labels = None

        if self.USE_CUDA:
            self.X_bow = self.X_bow.cuda()
            self.X_contextual = self.X_contextual.cuda()

        # Forward pass
        self.model.zero_grad()  # Update gradients to zero
        prior_mean, prior_variance, \
            posterior_mean, posterior_variance, posterior_log_variance, \
            word_dists, estimated_labels = self.model(
                self.X_bow, self.X_contextual, self.labels)

        # Backward pass: Compute gradients
        kl_loss, rl_loss = self._loss(self.X_bow, word_dists, prior_mean,
                                      prior_variance, posterior_mean,
                                      posterior_variance, posterior_log_variance)
        self.loss = self.weights["beta"] * kl_loss + rl_loss
        self.loss = self.loss.sum()
        
        if self.labels is not None:
            target_labels = torch.argmax(self.labels, 1)

            label_loss = torch.nn.CrossEntropyLoss()(estimated_labels, target_labels)
            loss += label_loss
            
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
    
    def deltaUpdateFit(
            self,
            modelStateDict: dict,
            save_dir: str,
            n_samples:int=20
        ) -> None:
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
        self.samples_processed += self.X_bow.size()[0]
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

    def optimize_on_minibatch_from_server(
            self,
            updates: dict
        ) -> None:
        # Update model's parameters from the forward pass carried at client's side
        self.model.topic_word_matrix = self.model.beta
        self.best_components = self.model.beta

        # Upadate gradients
        self.model.prior_mean.grad = torch.Tensor(updates["prior_mean"])
        self.model.prior_variance.grad = torch.Tensor(
            updates["prior_variance"])
        self.model.beta.grad = torch.Tensor(updates["beta"])

        # Perform one step of the optimizer (SGD/Adam)
        self.optimizer.step()

        return