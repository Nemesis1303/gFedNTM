
from abc import abstractmethod

from src.federation.federated_trainer_manager import FederatedTrainerManager


class FederatedModel(object):
    """
    Wrapper for a Generic Federated Topic Model. 
    """

    def __init__(self, tm_params, client, logger=None):

        self.tm_params = tm_params

        if logger:
            self.logger = logger
        else:
            import logging
            logging.basicConfig(level='INFO')
            self.logger = logging.getLogger('FederatedModel')

        self.fedTrManager = FederatedTrainerManager(client=client,
                                                    logger=self.logger)

    @abstractmethod
    def _train_minibatch(self):
        pass

    @abstractmethod
    def _optimize_on_minibatch(self):
        pass
    
    @abstractmethod
    def get_results_model(self):
        pass