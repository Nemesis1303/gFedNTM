# -*- coding: utf-8 -*-
"""
Created on Feb 1, 2022

@author: L. Calvo-Bartolomé (lcalvo@pa.uc3m.es)
"""

from src.utils.auxiliary_functions import (proto_to_modelStateDict,
                                           proto_to_optStateDict)


class FederatedTrainerManager(object):
    """

    """

    def __init__(self, client, logger=None):
        """
        Initilization Method

        Parameters
        ----------
        logger: Logger object
            To log object activity
        """

        self.client = client

        if logger:
            self._logger = logger
        else:
            import logging
            logging.basicConfig(level='INFO')
            self._logger = logging.getLogger('FederatedModel')

        return

    def send_gradient_minibatch(self, params):
        gradients_ = []
        for key in params.keys():
            if key not in ["current_mb", "current_epoch", "num_epochs"]:
                gradients_.append([key, params[key].grad.detach()])
        self.client.send_per_minibatch_gradient(
            gradients=gradients_,
            current_mb=params["current_mb"],
            current_epoch=params["current_epoch"],
            num_epochs=params["num_epochs"])

    def get_update_minibatch(self):

        # Wait until the server send the update
        request_update = self.client.listen_for_updates()

        # Initialize local_model with initial NN
        modelStateDict = proto_to_modelStateDict(
            request_update.nndata.modelUpdate)
        optStateDict = proto_to_optStateDict(request_update.nndata.optUpdate)

        return modelStateDict, optStateDict
