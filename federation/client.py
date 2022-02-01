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
import sys, getopt
import logging
import time
from timeloop import Timeloop
from datetime import timedelta
import grpc

import federated_pb2
import federated_pb2_grpc

##############################################################################
# DEBUG SETTINGS
##############################################################################
FORMAT = '[%(asctime)-15s] [%(filename)s] [%(levelname)s] %(message)s'
logging.basicConfig(format=FORMAT, level='INFO')
logger = logging.getLogger('client')

##############################################################################
#                                 CLIENT                                     #
##############################################################################
class Client:
    """[summary]
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
@looper.job(interval=timedelta(seconds=client.period))
def loop():
    id_message = "ID" + client.id + "_" + str(round(time.time()))
    header = federated_pb2.MessageHeader(id_request = id_message,
                                         message_type = federated_pb2.MessageType.CLIENT_TENSOR_SEND)
    metadata = federated_pb2.MessageAdditionalData(federation_completed = False,
                                                 iteration_completed = False,
                                                 current_iteration = 0,
                                                 num_max_iterations = 10)
    update_name = "Update of " + client.id
    data = federated_pb2.Update(tensor_name = update_name) # @TODO: Include gradient update
    request = federated_pb2.ClientTensorRequest(header = header, metadata = metadata, data = data)
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
            client.id = arg
        elif opt in ("-p", "--period"):
            client.period = arg
    
    with grpc.insecure_channel('localhost:50051') as channel:
        client.stub = federated_pb2_grpc.FederationStub(channel)
        looper.start(block=True)

if __name__ == '__main__':
    main(sys.argv[1:])