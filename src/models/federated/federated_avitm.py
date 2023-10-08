"""
Created on Feb 1, 2022
Last updated on Aug 24, 2023

.. codeauthor:: L. Calvo-Bartolomé (lcalvo@pa.uc3m.es)
"""
import datetime
import numpy as np
from src.models.base.pytorchavitm.avitm_network.avitm import AVITM
from src.models.federated.federated_model import FederatedModel
from src.utils.auxiliary_functions import convert_topic_word_to_init_size

class FederatedAVITM(AVITM, FederatedModel):
    """Class for the Federated AVITM model.
    """

    def __init__(
            self,
            tm_params: dict,
            grads_to_share: list[str],
            logger=None
        ) -> None:

        FederatedModel.__init__(self, tm_params, grads_to_share, logger)

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

        # Optimizer step
        self.optimizer.step()

        # Create gradient update to be sent to the server
        params = self.get_gradients()

        return params

    def deltaUpdateFit(
            self, 
            modelStateDict:dict, 
            save_dir:str, 
            n_samples:int=20
        ) -> None:
        """Updates gradient with aggregated gradient from server and calculates loss.

        Parameters
        ----------
        modelStateDict: dict
            Dictionary containing the model parameters to be updated.
        save_dir: str
            Directory where the model will be saved.
        n_samples: int
            Number of samples to be used for inference.
        """

        # Update local model's state dict
        self.set_gradients(modelStateDict)
        
        # Calculate minibatch's train loss and samples processed
        self.samples_processed += self.X_bow.size()[0]
        self.train_loss += self.loss.item()
        self.logger.info("-- -- Minibatch {} loss {} / samples processes {}".format(self.current_mb, self.loss.item(), self.samples_processed))

        # Minitbatch ends
        self.current_mb += 1

        try:
            # Get next minibatch
            self.current_batch_sample = next(self.train_loader_iter)
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
    # Evaluation
    # ======================================================
    def evaluate_synthetic_model(
            self,
            vocab_size: int,
            gt_thetas: np.array,
            gt_betas: np.array
        ) -> None:
        """Evaluates the model with the synthetic data.
        
        Parameters
        ----------
        vocab_size: int
            Size of the vocabulary.
        gt_thetas: np.array
            Ground truth doc-topic distribution.
        gt_betas: np.array
            Ground truth word-topic distribution.
        """

        # Get all vocabulary words (word1, word2, word3, etc.)
        all_words = \
            ['wd'+str(word) for word in np.arange(vocab_size+1)
             if word > 0] 
        
        # Convert word-topic distribution to the size of global vocabulary
        self.betas = convert_topic_word_to_init_size(
            vocab_size=vocab_size,
            model=self,
            model_type="avitm",
            ntopics=self.n_components,
            id2token=self.tm_params["id2token"],
            all_words=all_words)
        
        self.logger.info(
            f"-- -- Tópicos (equivalentes) evaluados correctamente: {np.sum(np.max(np.sqrt(self.betas).dot(np.sqrt(gt_betas.T)), axis=0))}")

        sim_mat_theoretical = \
            np.sqrt(gt_thetas[0]).dot(np.sqrt(gt_thetas[0].T))
        sim_mat_actual = np.sqrt(self.thetas).dot(np.sqrt(self.thetas.T))
        self.logger.info(
            f"-- -- Difference in evaluation of doc similarity: {np.sum(np.abs(sim_mat_theoretical - sim_mat_actual))/len(gt_thetas)}")
        
        return
