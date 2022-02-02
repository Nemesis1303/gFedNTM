# -*- coding: utf-8 -*-
"""
@author: lcalv
******************************************************************************
***                      CLASS FEDERATED SERVER                            ***
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
import numpy as np

import federated_pb2
import federated_pb2_grpc
import federation


class FederatedServer(federated_pb2_grpc.FederationServicer):
    """Class that describes the behaviour of the GRPC server to which several clients are connected to create a federation for the joint training of a CTM or ProdLDA model.

    """

    def __init__(self):
        self.federation = federation.Federation()
        self.last_update = None

    def record_client(self, context, id_client, gradient, current_iter, current_id_msg):
        """Method to record the communication between a server and one of the clients in the federation.

        Args:
            * context (AuthMetadataContext): An AuthMetadataContext providing information on the RPC
        """
        def unregister_client():
            """No-parameter callable to be called on RPC termination, that invokes the Federation's disconnect method when a client finishes a communication with the server.
            """
            self.federation.disconnect(context.peer(), id_client)
        context.add_callback(unregister_client)
        self.federation.connect(context.peer(), id_client, gradient, current_iter, current_id_msg)

    def sendLocalTensor(self, request, context):
        """[summary]

        Args:
            * request (ClientTensorRequest): [description]
            * context (AuthMetadataContext): [description]

        Returns:
            * ServerReceivedResponse: [description]
        """

        id_server = "IDS" + "_" + str(round(time.time()))
        header = federated_pb2.MessageHeader(id_response=id_server,
                                             id_to_request=request.header.id_request,
                                             message_type=federated_pb2.MessageType.SERVER_CONFIRM_RECEIVED)
        metadata = federated_pb2.MessageAdditionalData(federation_completed=request.metadata.federation_completed,
                                                       iteration_completed=request.metadata.iteration_completed,
                                                       current_iteration=request.metadata.current_iteration,
                                                       num_max_iterations=request.metadata.num_max_iterations)
        # TODO: Shape needs to be considered after deserialization. Make this convesrsion
        deserialized_bytes = np.frombuffer(request.data.tensor_content, dtype=np.int64)
        deserialized_numpy = np.reshape(deserialized_bytes, newshape=(2, 3))
        deserialized_tensor = torch.tensor(deserialized_numpy)
        print("The tensor sent is: ", deserialized_tensor)
        # deserialized_tensor = None
        self.record_client(context, request.metadata.id_machine, deserialized_tensor,
                           request.metadata.current_iteration, request.header.id_request)
       
        return federated_pb2.ServerReceivedResponse(header=header, metadata=metadata)


def serve():
    """[summary]
    """
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    federated_pb2_grpc.add_FederationServicer_to_server(
        FederatedServer(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    server.wait_for_termination()


if __name__ == '__main__':
    logging.basicConfig()
    serve()
