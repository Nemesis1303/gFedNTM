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

import federated_pb2
import federated_pb2_grpc

##############################################################################
# DEBUG SETTINGS
##############################################################################
FORMAT = '[%(asctime)-15s] [%(filename)s] [%(levelname)s] %(message)s'
logging.basicConfig(format=FORMAT, level='INFO')
logger = logging.getLogger('client')


class Client:
    """[summary]
    """

    def __init__(self, id, period):
        self.id = id
        self.period = period
        self.stub = None

    # looper = Timeloop()
    # @looper.job(interval=timedelta(seconds=client.period))
    # TODO: Change to take into account the number of iterations
    def getAverage(self):     
        id_message = "ID" + self.id + "_" + str(round(time.time()))
        header = federated_pb2.MessageHeader(id_request=id_message,
                                             message_type=federated_pb2.MessageType.CLIENT_TENSOR_SEND)
        metadata = federated_pb2.MessageAdditionalData(federation_completed=False,
                                                       iteration_completed=False,
                                                       current_iteration=0,
                                                       num_max_iterations=10,
                                                       id_machine = int(self.id))
        update_name = "Update of " + self.id
        content = torch.tensor(np.array([[3, 3, 3], [3, 3, 3]]))
        conent_bytes = content.numpy().tobytes()
        size = federated_pb2.TensorShape()
        size.dim.extend([federated_pb2.TensorShape.Dim(size=content.size(dim=0), name="dim1"),
                         federated_pb2.TensorShape.Dim(size=content.size(dim=1), name="dim2")])
        #dtype = str(content.numpy().dtype)
        #tensor_shape = size
        data = federated_pb2.Update(tensor_name=update_name,
                                    tensor_content=conent_bytes)
        request = federated_pb2.ClientTensorRequest(
            header=header, metadata=metadata, data=data)
        if self.stub:
            print(self.stub)
            response = self.stub.sendLocalTensor(request)
            logger.info('Client %s received a response to request %s',
                        str(self.id), response.header.id_to_request)


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

    with grpc.insecure_channel('localhost:50051') as channel:
        client.stub = federated_pb2_grpc.FederationStub(channel)
        # looper.start(block=True)
        client.getAverage()


if __name__ == '__main__':
    main(sys.argv[1:])
