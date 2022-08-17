import numpy as np
import torch

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
        self.client.send_per_minibatch_gradient(
                gradient = params["gradient"].grad.detach(),
                current_mb = params["current_mb"],
                current_epoch = params["current_epoch"],
                num_epochs = params["num_epochs"])
    
    def get_update_minibatch(self):

        # Wait until the server send the update
        request_update = self.client.listen_for_updates()

        # Update minibatch'gradient with the update from the server
        dims = tuple(
            [dim.size for dim in request_update.data.tensor_shape.dim])
        deserialized_bytes = np.frombuffer(
            request_update.data.tensor_content, dtype=np.float32)
        deserialized_numpy = np.reshape(
            deserialized_bytes, newshape=dims)
        deserialized_tensor = torch.Tensor(deserialized_numpy)
        
        return deserialized_tensor
    
    
    

    
