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
from __future__ import print_function
from importlib.metadata import metadata
import sys
import getopt
import logging
import time
from timeloop import Timeloop
from datetime import timedelta
import grpc
import torch
import numpy as np
import threading

import federated_pb2
import federated_pb2_grpc

##############################################################################
# DEBUG SETTINGS
##############################################################################
FORMAT = '[%(asctime)-15s] [%(filename)s] [%(levelname)s] %(message)s'
logging.basicConfig(format=FORMAT, level='INFO')
logger = logging.getLogger('client')


class Client:
    """ Class containing the stubs to interact with the server-side part via a gRPC channel.
    """

    def __init__(self, id, period):
        self.id = id
        self.period = period
        self.iter = 0
        with grpc.insecure_channel('localhost:50051') as channel:
            self.stub = federated_pb2_grpc.FederationStub(channel)
            content = torch.tensor(np.array([[3, 3, 3], [3, 3, 3]]))
            self.send_per_iteration_gradient(content, 0, 10)
            self.__listen_for_updates()
            
    # TODO: Change to take into account the number of iterations
    def send_per_iteration_gradient(self, gradient, iter, max_iter):     
        id_message = "ID" + self.id + "_" + str(round(time.time()))
        header = federated_pb2.MessageHeader(id_request=id_message,
                                             message_type=federated_pb2.MessageType.CLIENT_TENSOR_SEND)
        metadata = federated_pb2.MessageAdditionalData(current_iteration=iter,
                                                       num_max_iterations=max_iter,
                                                       id_machine = int(self.id))
        update_name = "Update of " + self.id
        conent_bytes = gradient.numpy().tobytes()
        size = federated_pb2.TensorShape()
        size.dim.extend([federated_pb2.TensorShape.Dim(size=gradient.size(dim=0), name="dim1"),
                         federated_pb2.TensorShape.Dim(size=gradient.size(dim=1), name="dim2")])
        data = federated_pb2.Update(tensor_name=update_name,
                                    tensor_shape = size,
                                    tensor_content=conent_bytes)
        request = federated_pb2.ClientTensorRequest(header=header, metadata=metadata, data=data)
        if self.stub:
            response = self.stub.sendLocalTensor(request)
            logger.info('Client %s received a response to request %s',
                        str(self.id), response.header.id_to_request)
    
    def __listen_for_updates(self):  
        update = self.stub.sendAggregatedTensor(federated_pb2.Empty())
        print("Client with ID ", self.id, "recevied update for iteration ", self.iter)
        self.iter +=1


# TODO: Maybe I am not interested on keeping a main here, but moving this to the init method and invoke several clients in an outside file
def main(argv):
    """[summary]

    Args:
        argv ([type]): [description]
    """    
    try:
        # TODO: Change to argparse
        opts, args = getopt.getopt(argv, "hn:p:", ["id=", "period="])
    except getopt.GetoptError:
        print('client.py -i <id> -p <period>')
        print('client.py --id <id> --period <period>')
        sys.exit(2)

    for opt, arg in opts:
        if opt == '-h':
            print('client.py -i <id> -p <period>')
            print('client.py --id <id> --period <period>')
            sys.exit()
        elif opt in ("-i", "--id"):
            client_id = arg
        elif opt in ("-p", "--period"):
            client_period = arg

    client = Client(client_id, client_period)

    #with grpc.insecure_channel('localhost:50051') as channel:
    #    client.stub = federated_pb2_grpc.FederationStub(channel)
        # create new listening thread for when new message streams come in
    #    threading.Thread(target=client.__listen_for_updates, daemon=True).start()
        #client.send_per_iteration_gradient()


if __name__ == '__main__':
    main(sys.argv[1:])
