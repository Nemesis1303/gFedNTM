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
# GENERAL IMPORTS
from concurrent import futures
import logging
import grpc
import time

# LOCAL IMPORTS
import federated_pb2
import federated_pb2_grpc


class FederatedServer(federated_pb2_grpc.FederationServicer):

    def sendLocalTensor(self, request, context):
        id_server = "IDS" + "_" + str(round(time.time()))
        header = federated_pb2.MessageHeader(id_response = id_server,
                                             id_to_request = request.header.id_request,
                                             message_type = federated_pb2.MessageType.SERVER_CONFIRM_RECEIVED)
        metadata = federated_pb2.MessageAdditionalData(federation_completed = request.metadata.federation_completed,
                                                    iteration_completed = request.metadata.iteration_completed,
                                                    current_iteration = request.metadata.current_iteration,
                                                    num_max_iterations = request.metadata.num_max_iterations)
        return federated_pb2.ServerReceivedResponse(header, metadata)


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    federated_pb2_grpc.add_FederationServicer_to_server(FederatedServer(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    server.wait_for_termination()


if __name__ == '__main__':
    logging.basicConfig()
    serve()