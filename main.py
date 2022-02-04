# -*- coding: utf-8 -*-
"""
@author: lcalv
******************************************************************************
***                                MAIN                                    ***
******************************************************************************
"""
##############################################################################
#                                IMPORTS                                     #
##############################################################################
from concurrent import futures
import grpc
import argparse

from federation import federated_pb2_grpc
from federation.client import Client
from federation.server import FederatedServer


def start_server(min_num_clients):
    # START SERVER
    server = grpc.server(futures.ThreadPoolExecutor())
    federated_pb2_grpc.add_FederationServicer_to_server(FederatedServer(min_num_clients), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    server.wait_for_termination()

def start_client(id_client):
    # TRAINING PARAMETERS
    period = 3
    training_datasets = ["aa","bb","cc","dd","ee"]
    model_parameters = {
        "input_size": 100,
        "n_components": 10,
        "model_type": "prodLDA",
        "hidden_sizes": (100, 100),
        "activation": "softplus",
        "dropout": 0.2,
        "learn_priors": True,
        "batch_size": 64,
        "lr": 2e-3,
        "momentum": 0.99,
        "solver": "adam",
        "num_epochs": 100,
        "reduce_on_plateau": False
    }

    # START CLIENT
    save_dir = None
    client = Client(id_client, period, model_parameters)
    client.train_local_model(training_datasets[id_client-1], save_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--id', type=int, default=0,
                        help="Id of the client. If this argument is not provided, the server is started.")
    parser.add_argument('--source', type=str, default=None,
                        help="Path to the training data.")
    parser.add_argument('--min_clients_federation', type=int, default=2,
                        help="Minimum number of client that are necessary for starting a federation. This parameter only affects the server.")
    args = parser.parse_args()

    if args.id == 0:
        print("Starting server with", args.min_clients_federation, "as minimum number of clients to start the federation.")
        start_server(args.min_clients_federation)
    else:
        print("Starting client with id ", args.id)
        start_client(args.id)

if __name__ == "__main__":
    main()










