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
import numpy as np

from federation import federated_pb2, federated_pb2_grpc, federation, federation_client
from utils.auxiliary_functions import get_type_from_string

FORMAT = '[%(asctime)-15s] [%(filename)s] [%(levelname)s] %(message)s'
logging.basicConfig(format=FORMAT, level='INFO')
logger = logging.getLogger('server')
GRPC_TRACE = all

# TODO: Include update client state


class FederatedServer(federated_pb2_grpc.FederationServicer):
    """Class that describes the behaviour of the GRPC server to which several clients are connected to create a federation for the joint training of a CTM or ProdLDA model.

    """

    def __init__(self, min_num_clients):
        self.federation = federation.Federation()
        self.min_num_clients = min_num_clients
        self.current_iteration = -1
        self.id_server = "IDS" + "_" + str(round(time.time()))

    def record_client(self, context, id_client, gradient,
                      current_iter, current_id_msg, num_max_iter):
        """Method to record the communication between a server and one of the clients in the federation.

        Args:
        -----
            * context (AuthMetadataContext): An AuthMetadataContext providing information on the RPC
        """
        def unregister_client():
            """No-parameter callable to be called on RPC termination, that invokes the Federation's disconnect method when a client finishes a communication with the server.
            """
            self.federation.disconnect(context.peer())
        context.add_callback(unregister_client)
        self.federation.connect(context.peer(), id_client,
                                gradient, current_iter, current_id_msg, num_max_iter)

    def record_client_waiting(self, context):
        def unregister_client():
            """No-parameter callable to be called on RPC termination, that invokes the Federation's disconnect method when a client finishes a communication with the server.
            """
            self.federation.disconnect(context.peer())
        context.add_callback(unregister_client)
        self.federation.connect_waiting(context.peer())

    def sendLocalTensor(self, request, context):
        """Sends an ACK response to the client after receiving his local tensor update.

        Args:
        -----
            * request (ClientTensorRequest): Request of the client for sending an iteration's 
                                             gradient
            * context (AuthMetadataContext): Context of the RPC communication between the 
                                             client and the server

        Returns:
        --------
            * ServerReceivedResponse: Server's response to confirm a client that its sent 
                                      gradient was received
        """

        header = federated_pb2.MessageHeader(id_response=self.id_server,
                                             id_to_request=request.header.id_request,
                                             message_type=federated_pb2.MessageType.SERVER_CONFIRM_RECEIVED)

        # Deserialize gradient update from the client and save in the corresponding Client object as numpy array
        dims = tuple([dim.size for dim in request.data.tensor_shape.dim])
        dtype_send = get_type_from_string(request.data.dtype)
        deserialized_bytes = np.frombuffer(
            request.data.tensor_content, dtype=dtype_send)
        deserialized_numpy = np.reshape(deserialized_bytes, newshape=dims)

        # Record client in the federation
        self.record_client(context, request.metadata.id_machine, deserialized_numpy,
                           request.metadata.current_epoch, request.header.id_request,
                           request.metadata.num_max_epochs)

        return federated_pb2.ServerReceivedResponse(header=header)

    def can_send_update(self):
        """Checks the conditions that need to be fullfilled in order to send the average tensor update to the clients.

        Returns:
        --------
            * boolean: True if the aggragate update can be sent.
        """

        if len(self.federation.federation_clients) < self.min_num_clients:
            return False
        # TODO: Check with condition can be added here in case not all clients sent the updates
        # if len(self.federation.federation) != len(self.federation.federation_clients):
        #    print("Not all the clients have sent their updates yet.")
        #    return False
        return True

    def sendAggregatedTensor(self, request, context):
        """Sends the average update to the correspinding client based on the context of the gRPC channel.

        Args:
        -----
            * request (federated_pb2.Empty): Empty protocol buffer coming from the client
            * context (AuthMetadataContext): Context of the RPC communication between the 
                                             client and the server

        Returns:
        --------
            federated_pb2.ServerAggregatedTensorRequest: Request with the aggregated tensor
        """

        # Record clients waiting
        self.record_client_waiting(context)

        # Wait until all the clients in the federation have sent its current iteration gradient
        wait(lambda: self.can_send_update(), timeout_seconds=120,
             waiting_for="Update can be sent")

        # Calculate average
        clients_tensors = [
            client.tensor for client in self.federation.federation_clients]
        average_tensor = np.mean(np.stack(clients_tensors), axis=0)
        print("The average tensor is: ", average_tensor)

        # Serialize tensor
        content_bytes = average_tensor.tobytes()
        size = federated_pb2.TensorShape()
        num_dims = len(average_tensor.shape)
        for i in np.arange(num_dims):
            name = "dim" + str(i)
            size.dim.extend(
                [federated_pb2.TensorShape.Dim(size=average_tensor.shape[i], name=name)])

        # Get the client to sent the update to based on the context of the gRPC channel
        client_to_repond = \
         federation_client.FederationClient.get_pos_by_key(context.peer(),
                                                           self.federation.federation_clients)
        # Create request
        update_name = "Update for client with ID " + \
         str(self.federation.federation_clients[client_to_repond].id) + \
         " for the iteration " + \
         str(self.federation.federation_clients[client_to_repond].current_iter)

        data = federated_pb2.Update(tensor_name=update_name,
                                    tensor_shape=size, tensor_content=content_bytes)
        # Send update
        header = federated_pb2.MessageHeader(id_response=self.id_server,
                                             message_type=federated_pb2.MessageType.SERVER_AGGREGATED_TENSOR_SEND)
        print(update_name)
        return federated_pb2.ServerAggregatedTensorRequest(header=header, data=data)
