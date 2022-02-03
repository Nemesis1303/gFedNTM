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
from waiting import wait
import logging
import grpc
import time
import torch
import threading
import numpy as np
import pdb

import federated_pb2
import federated_pb2_grpc
import federation

FORMAT = '[%(asctime)-15s] [%(filename)s] [%(levelname)s] %(message)s'
logging.basicConfig(format=FORMAT, level='INFO')
logger = logging.getLogger('server')
GRPC_TRACE=all 

class FederatedServer(federated_pb2_grpc.FederationServicer):
    """Class that describes the behaviour of the GRPC server to which several clients are connected to create a federation for the joint training of a CTM or ProdLDA model.

    """

    def __init__(self):
        self.federation = federation.Federation()
        self.last_update = None
        self.min_num_clients = 2 # TODO: Increase
        self.current_iteration = -1
        self.id_server = "IDS" + "_" + str(round(time.time()))

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
            * request (ClientTensorRequest): Request of the client for sending an iteration's gradient
            * context (AuthMetadataContext): Context of the RPC communication between the client and the server

        Returns:
            * ServerReceivedResponse: Server's response to confirm a client that its sent gradient was received
        """

        header = federated_pb2.MessageHeader(id_response=self.id_server,
                                             id_to_request=request.header.id_request,
                                             message_type=federated_pb2.MessageType.SERVER_CONFIRM_RECEIVED)
        dims = tuple([dim.size for dim in request.data.tensor_shape.dim])
        deserialized_bytes = np.frombuffer(request.data.tensor_content, dtype=np.float64)
        deserialized_numpy = np.reshape(deserialized_bytes, newshape=dims)    

        self.record_client(context, request.metadata.id_machine, deserialized_numpy,
                           request.metadata.current_iteration, request.header.id_request)
       
        return federated_pb2.ServerReceivedResponse(header=header)
    
    def can_send_update(self):
        #for client in self.federation.federation_clients:
        #    if self.current_iteration != client.current_iter:
        print(len(self.federation.federation_clients) >= self.min_num_clients)
        if len(self.federation.federation_clients) < self.min_num_clients:
            return False
        return True
    
    def sendAggregatedTensor(self, request, context):
        print("llega al servidor")
        print(self.federation.federation_clients[0])
        
        # Wait until all the clients in the federation have sent its current iteration gradient
        wait(lambda: self.can_send_update(), timeout_seconds=120, waiting_for="Update can be sent")
        # Calculate average
        clients_tensors = [client.tensor for client in self.federation.federation_clients]
        average_tensor = np.mean(np.stack(clients_tensors), axis=0)
        print("The average tensor is: ", average_tensor)
        # Serialize tensor
        content_bytes = average_tensor.tobytes()
        size = federated_pb2.TensorShape()
        size.dim.extend([federated_pb2.TensorShape.Dim(size=average_tensor.shape[0], name="dim1"),
                        federated_pb2.TensorShape.Dim(size=average_tensor.shape[1], name="dim2")])
        update_name = "Update from iteration " 
        print(update_name)
        data = federated_pb2.Update(tensor_name=update_name, 
                                    tensor_shape=size, tensor_content=content_bytes)
        # Send update
        # TODO: Check id server
        id_server = "IDS" + "_" + str(round(time.time()))
        print(id_server)
        #id_to_request=request.header.id_request,
        header = federated_pb2.MessageHeader(id_response=id_server,
                                            message_type=federated_pb2.MessageType.SERVER_AGGREGATED_TENSOR_SEND)
        print("Server sends update")
        return federated_pb2.ServerAggregatedTensorRequest(header=header, data=data)



def serve():
    """[summary]
    """
    server = grpc.server(futures.ThreadPoolExecutor())
    federated_pb2_grpc.add_FederationServicer_to_server(
        FederatedServer(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    
    server.wait_for_termination()


if __name__ == '__main__':
    logging.basicConfig()
    serve()
