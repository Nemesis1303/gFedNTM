"""
Created on Feb 1, 2022
Last updated on Aug 24, 2023

.. codeauthor:: L. Calvo-Bartolomé (lcalvo@pa.uc3m.es)
"""
from abc import abstractmethod
from typing import Union
import numpy as np
from torch.utils.data import DataLoader
from src.models.base.pytorchavitm.datasets.bow_dataset import BOWDataset
from src.models.base.contextualized_topic_models.datasets.dataset import CTMDataset
from src.utils.auxiliary_functions import save_model_as_npz
from sklearn.preprocessing import normalize

class FederatedModel(object):
    """
    Wrapper for a Generic Federated Topic Model. 
    """

    def __init__(
            self,
            tm_params: dict,
            logger=None
        ) -> None:

        self.tm_params = tm_params

        if logger:
            self.logger = logger
        else:
            import logging
            logging.basicConfig(level='INFO')
            self.logger = logging.getLogger('FederatedModel')
        
        # Parameters for tracking federated model
        self.model_dir = None
        self.train_data = None
        self.current_mb = -1
        self.current_epoch = -1
        self.samples_processed = -1
        self.train_loader_iter = None
        self.current_batch_sample = None

        # Post-training parameters
        self.topics = None
        self.thetas = None
        self.betas = None

    # ======================================================
    # Client-side training
    # ======================================================
    def preFit(
            self,
            train_data: Union[BOWDataset,CTMDataset],
            save_dir=None
        ) -> None:
        """Carries out the initialization of all parameters needed for training of a local model.

        Parameters
        ----------
        train_data: Union[BOWDataset,CTMDataset]
            Training dataset.
        save_dir: str, optional
            Directory to save checkpoint models to, defaults to None.
        """

        self.model_dir = save_dir
        self.train_data = train_data

        # Initialize training variables before the training loop
        self.current_mb = 0
        self.current_epoch = 0
        self.samples_processed = 0
        self.train_loss = 0

        # Initialize train dataLoader and get first sample
        self.train_loader = DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_data_loader_workers)
        self.train_loader_iter = iter(self.train_loader)
        self.current_batch_sample = next(self.train_loader_iter)

        # Set the model to train mode before starting the first epoch
        # This should be done at the beginning of each epoch
        self.model.train()

        # Training of the local model starts ->>>

        return

    @abstractmethod
    def train_mb_delta(self):
        pass

    @abstractmethod
    def deltaUpdateFit(self):
        pass

    # ======================================================
    # Server-side training
    # ======================================================
    @abstractmethod
    def optimize_on_minibatch_from_server(self):
        pass

    # ======================================================
    # Evaluation
    # ======================================================
    def get_results_model(
            self,
            save_dir:str
        ) -> None:
        """Gets the results of the model after training at the CLIENT side.
        
        Parameters
        ----------
        save_dir: str
            Directory where the model will be saved.
        """

        # Get topics
        self.topics = self.get_topics()
        print(self.topics)

        # Get doc-topic distribution
        self.thetas = \
            np.asarray(self.get_doc_topic_distribution(self.train_data))
        self.thetas[self.thetas < 3e-3] = 0
        self.thetas = normalize(self.thetas, axis=1, norm='l1')

        # Get word-topic distribution
        self.betas = self.get_topic_word_distribution()
        print(self.betas)

        # Save model
        self.logger.info(f"-- -- Saving model at {save_dir} ")
        save_model_as_npz(save_dir, self)
    
    def get_topics_in_server(
            self,
            save_dir:str
        ) -> None:
        """Gets the topics of the model after training at the SERVER side.
        The topics' chemical description cannot be inferred since the server does not have access to the training corpus. Inference is required in order to get the topic distribution.
        """
        
        # Get word-topic distribution
        self.betas = self.get_topic_word_distribution()
        
        self.logger.info(f"-- -- Saving global model...")
        save_model_as_npz(save_dir, self)
        
        return
