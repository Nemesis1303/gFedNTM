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
from importlib.metadata import metadata
import sys, getopt
import logging
import time
from __future__ import print_function
from timeloop import Timeloop
from datetime import timedelta
import grpc

# LOCAL IMPORTS
import federated_pb2
import federated_pb2_grpc


FORMAT = '[%(asctime)-15s] [%(filename)s] [%(levelname)s] %(message)s'
logging.basicConfig(format=FORMAT, level='INFO')
logger = logging.getLogger('client')

##############################################################################
#                                 CLIENT                                     #
##############################################################################

class Client:
    """
    Attributes:
    ----------
        * id
        * period
        * stub
    """
    def __init__(self, id, period):
        self.id = id
        self.period = period
        self.stub = None

client = Client("client", 4)

##############################################################################
#                                 LOOPER                                     #
##############################################################################
looper = Timeloop()
#@looper.job(interval=timedelta(seconds=client.period))
@looper.job(interval=timedelta(seconds=4))
def loop():
    id_message = "ID" + str(client.id) + "_" + str(round(time.time()))
    header = federated_pb2.MessageHeader(id_request = id_message,
                                         message_type = federated_pb2.MessageType.CLIENT_TENSOR_SEND)
    metada = federated_pb2.MessageAdditionalData(federation_completed = False,
                                                 iteration_completed = False,
                                                 current_iteration = 0,
                                                 num_max_iterations = 10)
    gradient_name = "Gradient of " + str(client.id)
    data = federated_pb2.Update(gradientName = gradient_name) # @TODO: Include gradient update
    request = federated_pb2.ClientTensorRequest(header, metadata, data)
    response = client.stub.sendLocalTensor(request)
    logger.info('Client %s received a response to request %s', str(client.id), response.header.id_to_request)


##############################################################################
#                                 MAIN                                       #
##############################################################################
def main(argv):
    try:
        opts, args = getopt.getopt(argv,"hn:p:",["id=","period="])
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
            client.name = arg
        elif opt in ("-p", "--period"):
            client.period = arg
    
    # NOTE(gRPC Python Team): .close() is possible on a channel and should be
    # used in circumstances in which the with statement does not fit the needs
    # of the code.
    with grpc.insecure_channel('localhost:50051') as channel:
        client.stub = federated_pb2_grpc.FederationStub(channel)
        looper.start(block=True)

if __name__ == '__main__':
    main(sys.argv[1:])