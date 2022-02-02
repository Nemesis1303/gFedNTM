# -*- coding: utf-8 -*-
"""
@author: lcalv
******************************************************************************
***                             CLASS CLIENT                               ***
******************************************************************************
"""
##############################################################################
#                                IMPORTS                                     #
##############################################################################
from concurrent import futures
import logging
import grpc
import time
import torch 
import threading

import federated_pb2
import federated_pb2_grpc
import federation

##############################################################################
#                          FEDERATED SERVEER                                 #
##############################################################################
class FederatedServer(federated_pb2_grpc.FederationServicer):
    """[summary]

    Args:
        federated_pb2_grpc ([type]): [description]

    Returns: 
        [type]: [description]
    """    """
    Attributes:
    ----------
        * 
    """
    def __init__(self):
        self.federation = federation.Federation()
    
    def record_client(self, context):
        def unregister_client():
            self.federation.disconnect(context.peer())
        context.add_callback(unregister_client)
        self.federation.connect(context.peer())

    def sendLocalTensor(self, request, context):
        """[summary]

        Args:
            request ([ClientTensorRequest]): [description]
            context ([AuthMetadataContext]): [description]

        Returns:
            [type]: [description]
        """        

        self.record_client(context)
        print("Clients: {}".format(self.federation.getClients()))

        id_server = "IDS" + "_" + str(round(time.time()))
        header = federated_pb2.MessageHeader(id_response = id_server,
                                             id_to_request = request.header.id_request,
                                             message_type = federated_pb2.MessageType.SERVER_CONFIRM_RECEIVED)
        metadata = federated_pb2.MessageAdditionalData(federation_completed = request.metadata.federation_completed,
                                                       iteration_completed = request.metadata.iteration_completed,
                                                       current_iteration = request.metadata.current_iteration,
                                                       num_max_iterations = request.metadata.num_max_iterations)
        # TODO: Shape needs to be considered after deserialization
        # deserialized_bytes = np.frombuffer(bytes, dtype=np.int8)
        # deserialized_x = np.reshape(deserialized_bytes, newshape=(2, 2))
        # deserialized_tensor = torch.tensor(np.reshape(np.frombuffer(request.data.tensor_content, dytype=np.int64),newshape=(3, 3)))
        # print("The tensor sent is: ", deserialized_tensor)
        return federated_pb2.ServerReceivedResponse(header = header, metadata = metadata)


def serve():
    """[summary]
    """    
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    federated_pb2_grpc.add_FederationServicer_to_server(FederatedServer(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    server.wait_for_termination()


if __name__ == '__main__':
    logging.basicConfig()
    serve()