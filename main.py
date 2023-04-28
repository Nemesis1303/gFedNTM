# -*- coding: utf-8 -*-
"""
Main script to run the federated topic model.

Created on Feb 4, 2022
@author: L. Calvo-Bartolomé (lcalvo@pa.uc3m.es)
"""

import argparse
import multiprocessing as mp
import pathlib
from concurrent import futures
from subprocess import check_output

import grpc
import numpy as np
import pandas as pd

from src.federation.client import Client, FederatedClientServer
from src.federation.server import FederatedServer
from src.protos import federated_pb2_grpc

address = ['localhost:50051', 'gfedntm-server:50051']
# ======================================================
# Training params
# ======================================================
fixed_parameters = {
    "contextual_size": 192,  # 768
    "inference_type": 'combined',
    "n_components": 10,
    "model_type": 'prodLDA',
    "hidden_sizes": (100, 100),
    "activation": 'softplus',
    "dropout": 0.2,
    "learn_priors": True,
    "lr": 2e-3,
    "momentum": 0.99,
    "solver": 'adam',
    "num_epochs": 5,
    "batch_size": 4,
    "num_samples": 10,
    "reduce_on_plateau": False,
    "num_data_loader_workers": mp.cpu_count(),
    "label_size": 0,
    "loss_weights": None,
    "topic_prior_mean": 0.0,
    "topic_prior_variance": None,
    "verbose": True
}

tuned_parameters = {
    "bow_size": 0,
    "input_size": 0,
    "n_components": 50,
}


def start_server(min_num_clients, model_type):
    """Initializes the server that is going to orchestrates the federated training.

    Parameters
    ----------
    min_num_clients : int
        Minimum number of clients to start the federation
    model_type: str
        Underlying topic modeling algorithm with which the federated topic model is going to be constructed (prod|ctm)
    """

    # START SERVER
    opts = [("grpc.keepalive_time_ms", 10000),
            ("grpc.keepalive_timeout_ms", 5000),
            ("grpc.keepalive_permit_without_calls", True),
            ("grpc.http2.max_ping_strikes", 0)]

    server = grpc.server(futures.ThreadPoolExecutor(), options=opts)
    federated_server = FederatedServer(
        min_num_clients=min_num_clients,
        model_params={**fixed_parameters, **tuned_parameters}, model_type=model_type
    )

    federated_pb2_grpc.add_FederationServicer_to_server(
        federated_server, server)
    server.add_insecure_port('[::]:50051')
    server.start()
    server.wait_for_termination()

def start_client(id_client: int, data_type: str, fos: str, source: str):
    """Initialize a client that is going to contribute to the training of a federated topic model.

    Parameters
    ----------
    id_client : int
        Client's identifier
    data_type : str
        Type of the data that is going to be used by the client for the training (synthetic|real)
    fos : str
        Category or label describing the data in source belonging to the client given by 'id_client'
    source: str
        Path to the data that is going to be used by the client for the training
    """

    if data_type == "synthetic":
        data = np.load(source, allow_pickle=True)
        corpus = data['documents'][id_client-1]
        vocab_size = data['vocab_size']
        word_topic_distrib_gt = data['topic_vectors']
        doc_topic_distrib_gt_all = data['doc_topics']
        doc_topic_distrib_gt_together = []
        for i in np.arange(len(doc_topic_distrib_gt_all)):
            doc_topic_distrib_gt_together.extend(doc_topic_distrib_gt_all[i])

    elif data_type == "real":
        corpusFile = pathlib.Path(source)
        df = pd.read_parquet(corpusFile)
        df = df[df['fos'] == fos]
        corpus = df

    else:
        print("Specified data type not supported")

    # START CLIENT
    # Open channel for communication with the server
    MAX_MESSAGE_LENGTH = 80 * 1024 * 1024
    MAX_INBOUND_MESSAGE_SIZE = 8 * 1024 * 1024
    MAX_INBOUND_METADATA_SIZE = 80 * 1024 * 1024
    options = [
        ('grpc.max_message_length', MAX_MESSAGE_LENGTH),
        ('grpc.max_send_message_length', MAX_MESSAGE_LENGTH),
        ('grpc.max_receive_message_length', MAX_MESSAGE_LENGTH),
        ('grpc.max_inbound_message_size', MAX_INBOUND_MESSAGE_SIZE),
        ('grpc.max_inbound_metadata_size', MAX_INBOUND_METADATA_SIZE),
        ('grpc.max_metadata_size', MAX_INBOUND_METADATA_SIZE)
    ]
    with grpc.insecure_channel(address[0], options=options) as channel:
        stub = federated_pb2_grpc.FederationStub(channel)

        # Create client
        client = Client(id=id_client,
                        stub=stub,
                        local_corpus=corpus,
                        data_type=data_type)

        # # Start training of local model
        # client.train_local_model()

        # # Print evaluation results if data_type is synthetic
        # if data_type == "synthetic":
        #     eval_parameters = [
        #         vocab_size, doc_topic_distrib_gt_all[id_client-1], word_topic_distrib_gt]
        #     client.eval_local_model(eval_params=eval_parameters)

    # Start client-server
    # opts = [("grpc.keepalive_time_ms", 10000),
    #         ("grpc.keepalive_timeout_ms", 5000),
    #         ("grpc.keepalive_permit_without_calls", True),
    #         ("grpc.http2.max_ping_strikes", 0)]

    # client_server = grpc.server(futures.ThreadPoolExecutor(), options=opts)
    # federated_server = FederatedClientServer(
    #     client.local_model, client.train_data)
    # federated_pb2_grpc.add_FederationServerServicer_to_server(
    #     federated_server, client_server)
    # client_server.add_insecure_port('[::]:' + str(50051 + id_client))
    # client_server.start()
    # client_server.wait_for_termination()


def preproc(spark: bool, nw: int, configFile: pathlib.Path):
    """Carries out simple preprocessing tasks and prepare a training file in the format required by the federation.
    """
    if spark:
        # Run command for corpus preprocessing using Spark
        script_spark = "/export/usuarios_ml4ds/lbartolome/spark/script-spark"
        token_spark = "/export/usuarios_ml4ds/lbartolome/spark/tokencluster.json"
        script_path = './src/preprocessing/text_preproc.py'
        machines = str(10)
        cores = str(4)
        options = '"--spark --config ' + configFile.resolve().as_posix() + '"'
        cmd = script_spark + ' -C ' + token_spark + \
            ' -c ' + cores + ' -N ' + machines + ' -S ' + script_path + ' -P ' + options
        print(cmd)
        try:
            print(f'-- -- Running command {cmd}')
            output = check_output(args=cmd, shell=True)
        except:
            print('-- -- Execution of script failed')
    else:
        # Run command for corpus preprocessing using gensim
        # Preprocessing will be accelerated with Dask using the number of
        # workers indicated in the configuration file for the project
        cmd = f'python src/preprocessing/text_preproc.py --config {configFile.as_posix()} --nw {nw}'

        try:
            print(f'-- -- Running command {cmd}')
            output = check_output(args=cmd, shell=True)
        except:
            print('-- -- Execution of script failed')


def main():
    parser = argparse.ArgumentParser()

    # ========================================================================
    # Parameters for trainings (if --preproc not specified, training happens)
    # ========================================================================
    #####################
    # Client parameters #
    #####################
    parser.add_argument('--id', type=int, default=0,
                        help="Client ID. \
                        If not provided, the server (ID=0) is started.")
    parser.add_argument('--source', type=str,
                        default="/workspaces/gFedNTM/static/datasets/synthetic.npz",
                        help="Path to the training data.")
    parser.add_argument('--data_type', type=str, default="synthetic",
                        help="synthetic or real")
    parser.add_argument("--fos", type=str, default="computer_science",
                        help="Category")
    #####################
    # Server parameters #
    #####################
    parser.add_argument('--min_clients_federation', type=int, default=1,
                        help="Minimum number of client that are necessary for starting a federation. This parameter only affects the server.")
    parser.add_argument('--model_type', type=str, default="avitm",
                        help="Underlying model type: avitm/ctm.")

    # ========================================================================
    # Parameters for preprocessing
    # ========================================================================
    parser.add_argument('--preproc', action='store_true', default=False,
                        help="Selects preprocessing mode.")
    parser.add_argument('--spark', action='store_true', default=False,
                        help='Indicate that spark cluster is available',
                        required=False)
    parser.add_argument('--nw', type=int, required=False, default=0,
                        help="Number of workers when preprocessing data with Dask. Use 0 to use Dask default")
    parser.add_argument('--config', type=str, default=None,
                        help="path to configuration file")

    args = parser.parse_args()

    if args.preproc:
        print("Preprocessing data")
        preproc(spark=args.spark, nw=args.nw,
                configFile=pathlib.Path(args.config))
    elif args.id == 0:
        print("Starting server with", args.min_clients_federation,
              "as minimum number of clients to start the federation.")
        start_server(args.min_clients_federation, args.model_type)
    else:
        print("Starting client with id ", args.id)
        start_client(args.id, args.data_type, args.fos, args.source)


if __name__ == "__main__":
    main()
