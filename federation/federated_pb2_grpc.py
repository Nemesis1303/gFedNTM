# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc

import federated_pb2 as federated__pb2


class FederationStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.sendLocalTensor = channel.unary_unary(
                '/federated.Federation/sendLocalTensor',
                request_serializer=federated__pb2.ClientTensorRequest.SerializeToString,
                response_deserializer=federated__pb2.ServerReceivedResponse.FromString,
                )
        self.sendAggregatedTensor = channel.unary_unary(
                '/federated.Federation/sendAggregatedTensor',
                request_serializer=federated__pb2.ServerAggregatedTensorRequest.SerializeToString,
                response_deserializer=federated__pb2.ClientReceivedResponse.FromString,
                )


class FederationServicer(object):
    """Missing associated documentation comment in .proto file."""

    def sendLocalTensor(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def sendAggregatedTensor(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_FederationServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'sendLocalTensor': grpc.unary_unary_rpc_method_handler(
                    servicer.sendLocalTensor,
                    request_deserializer=federated__pb2.ClientTensorRequest.FromString,
                    response_serializer=federated__pb2.ServerReceivedResponse.SerializeToString,
            ),
            'sendAggregatedTensor': grpc.unary_unary_rpc_method_handler(
                    servicer.sendAggregatedTensor,
                    request_deserializer=federated__pb2.ServerAggregatedTensorRequest.FromString,
                    response_serializer=federated__pb2.ClientReceivedResponse.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'federated.Federation', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


 # This class is part of an EXPERIMENTAL API.
class Federation(object):
    """Missing associated documentation comment in .proto file."""

    @staticmethod
    def sendLocalTensor(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/federated.Federation/sendLocalTensor',
            federated__pb2.ClientTensorRequest.SerializeToString,
            federated__pb2.ServerReceivedResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def sendAggregatedTensor(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/federated.Federation/sendAggregatedTensor',
            federated__pb2.ServerAggregatedTensorRequest.SerializeToString,
            federated__pb2.ClientReceivedResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)
