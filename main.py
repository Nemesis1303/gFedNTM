# -*- coding: utf-8 -*-
"""
@author: lcalv
******************************************************************************
***                                MAIN                                    ***
******************************************************************************
"""
import argparse
import multiprocessing as mp
from concurrent import futures

import grpc
import numpy as np

from federation import federated_pb2_grpc
from federation.client import Client
from federation.server import FederatedServer
from utils.utils_preprocessing import prepare_data_avitm_federated


def start_server(min_num_clients):
    # START SERVER
    server = grpc.server(futures.ThreadPoolExecutor())
    federated_pb2_grpc.add_FederationServicer_to_server(
        FederatedServer(min_num_clients), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    server.wait_for_termination()


def start_client(id_client, model_type):
    # Training data
    file = "data/training_data/synthetic.npz"
    data = np.load(file, allow_pickle=True)
    corpus = data['documents'][id_client-1]
    vocab_size = data['vocab_size']
    word_topic_distrib_gt = data['topic_vectors']
    doc_topic_distrib_gt_all = data['doc_topics']
    doc_topic_distrib_gt_together = []
    for i in np.arange(len(doc_topic_distrib_gt_all)):
        doc_topic_distrib_gt_together.extend(doc_topic_distrib_gt_all[i])

    # Generate training dataset in the format for AVITM
    train_dataset, input_size, id2token = prepare_data_avitm_federated(
        corpus, 0.99, 0.01)

    if model_type == "ctm":
        model_parameters = {
            "bow_size": input_size,
            "contextual_size": 10,
            "inference_type": "combined",
            "n_components": 10,
            "model_type": "prodLDA",
            "hidden_sizes": (100, 100),
            "activation": 'softplus',
            "dropout": 0.2,
            "learn_priors": True,
            "lr": 2e-3,
            "momentum": 0.99,
            "solver": 'adam',
            "num_epochs": 100,
            "reduce_on_plateau": False,
            "num_data_loader_workers": mp.cpu_count(),
            "label_size": 0,
            "loss_weights": None
        }

    elif model_type == "prod":

        # TRAINING PARAMETERS
        model_parameters = {
            "input_size": input_size,
            "n_components": 50,
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
            "num_samples": 10,
            "num_data_loader_workers": 0,
            "reduce_on_plateau": False,
            "topic_prior_mean": 0.0,
            "topic_prior_variance": None,
            "verbose": True
        }

        # START CLIENT
        # Open channel for communication with the server
        with grpc.insecure_channel('localhost:50051') as channel:
            stub = federated_pb2_grpc.FederationStub(channel)
            client = Client(id_client, stub, model_type, corpus, model_parameters)
            client.train_local_model()

    else:
        print("The selected model type is not provided.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--id', type=int, default=0,
                        help="Id of the client. If this argument is not provided, the server is started.")
    parser.add_argument('--source', type=str, default=None,
                        help="Path to the training data.")
    parser.add_argument('--model_type', type=str, default="prod",
                        help="ProdLDA (prod) or CTM (ctm).")
    parser.add_argument('--min_clients_federation', type=int, default=1,
                        help="Minimum number of client that are necessary for starting a federation. This parameter only affects the server.")
    args = parser.parse_args()

    if args.id == 0:
        print("Starting server with", args.min_clients_federation,
              "as minimum number of clients to start the federation.")
        start_server(args.min_clients_federation)
    else:
        print("Starting client with id ", args.id)
        start_client(args.id, args.model_type)


if __name__ == "__main__":
    main()
