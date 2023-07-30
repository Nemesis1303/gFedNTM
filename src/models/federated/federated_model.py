"""
Created on Feb 1, 2022

@author: L. Calvo-Bartolomé (lcalvo@pa.uc3m.es)
"""
from abc import abstractmethod

class FederatedModel(object):
    """
    Wrapper for a Generic Federated Topic Model. 
    """

    def __init__(self,
                 tm_params: dict,
                 #client: Client,
                 logger=None) -> None:

        self.tm_params = tm_params

        if logger:
            self.logger = logger
        else:
            import logging
            logging.basicConfig(level='INFO')
            self.logger = logging.getLogger('FederatedModel')

        #self.fedTrManager = FederatedTrainerManager(client=client,
        #                                           logger=self.logger)

    # ======================================================
    # Client-side training
    # ======================================================
    @abstractmethod
    def preFit(self):
        pass

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
    @abstractmethod
    def get_results_model(self):
        pass
