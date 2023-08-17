# -*- coding: utf-8 -*-
"""
Main script to run the federated topic model.

Created on Feb 4, 2022
@author: L. Calvo-Bartolomé (lcalvo@pa.uc3m.es)
"""

import argparse
import configparser
import os
import pathlib
from concurrent import futures
import datetime as DT
import grpc
import numpy as np
import pandas as pd

from src.federation.client import Client
from src.federation.server import FederatedServer
from src.protos import federated_pb2_grpc
from src.utils.auxiliary_functions import read_config_experiments

def start_server(min_num_clients:int,
                 model_type:str,
                 max_iters:int,
                 opts_server:list,
                 opts_client:list,
                 address:str,
                 training_params:dict,
                 save_server:str,
                 logs_server:str,
                 server_port:int=50051,
                 time_termination:int=1200):
    """Initializes the server that is going to orchestrates the federated training.

    Parameters
    ----------
    min_num_clients : int
        Minimum number of clients to start the federation
    model_type: str
        Underlying topic modeling algorithm with which the federated topic model is going to be constructed (prod|ctm)
    max_iters: int
        Maximum number of iterations for the federated training
    opts_server: list
        List of options for the server
    opts_client: list
        List of options for the server when it behaves as a 'client'
    address: str
        Address of the server
    training_params: dict
        Dictionary with the parameters for the training of the federated topic model
    save_server: str
        Path to save the global model of the federated topic model
    logs_server: str
        Path to save the logs of the server
    server_port: int
        Port where the server is going to be listening
    time_termination : int
        Time in seconds after which the server is going to be terminated if no client has connected to it
    """
    
    client_server_addres = \
        "gfedntm-client" if address.startswith("gfedntm-server") else "localhost"
    print(client_server_addres)
    
    save_server += f"/global_model_{DT.datetime.now().strftime('%Y%m%d')}"
    logs_server += f"/logs_{DT.datetime.now().strftime('%Y%m%d')}.txt"
    server = grpc.server(futures.ThreadPoolExecutor(), options=opts_server)
    federated_server = FederatedServer(
                            min_num_clients=min_num_clients,
                            model_params={**training_params},
                            model_type=model_type,
                            max_iters=max_iters,
                            opts_client=opts_client,
                            save_server=save_server,
                            logs_server=logs_server,
                            client_server_addres=client_server_addres)

    federated_pb2_grpc.add_FederationServicer_to_server(
        federated_server, server)
    server_port = '[::]:' + str(server_port)
    server.add_insecure_port(server_port)
    server.start()
    server.wait_for_termination(time_termination)

def start_client(id_client:int,
                 data_type:str,
                 fos:str,
                 source:str,
                 address:str,
                 opts_client:list,
                 opts_server:list,
                 save_client:str,
                 logs_client:str):
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
    address: str
        Address of the server
    opts_client: list
        List of options for the client
    opts_server: list
        List of options for the 'client-server'
    save_client: str
        Path to the folder where the client is going to save the results of the training
    logs_client: str
        Path to the file where the client is going to save the logs of the training
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
        
    save_client += f"{id_client}/model_{id_client}_{DT.datetime.now().strftime('%Y%m%d')}"
    logs_client += f"{id_client}/logs_{DT.datetime.now().strftime('%Y%m%d')}.txt"

    # START CLIENT
    # Open channel for communication with the server
    with grpc.insecure_channel(address, options=opts_client) as channel:
        stub = federated_pb2_grpc.FederationStub(channel)

        # Create client
        _ = Client(id=id_client,
                   stub=stub,
                   local_corpus=corpus,
                   data_type=data_type,
                   opts_server=opts_server,
                   save_client=save_client,
                   logs_client=logs_client)

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
    parser.add_argument('--max_iters', type=int, default=100,
                        help="Maximum number of global iterations to train the federated topic model.")
    args = parser.parse_args()
    
    # Read default training parameters
    #workdir =  os.path.dirname(os.path.dirname(os.getcwd()))
    workdir = "/workspace" # TODO: Configure
    configFile = os.path.join(workdir, "config/dft_params.cf")
    training_params = read_config_experiments(configFile)

    # Define address for communication between client and server
    config = configparser.ConfigParser()
    config.read(configFile)
    if workdir.startswith("/workspace/gFedNTM/") or not workdir.startswith("/workspace"):
        address = config.get("addresses", "local")
    else:
        address = config.get("addresses", "docker")
    print(address)
    
    # Define directories to save results
    save_client = config.get("save_dir", "save_client")
    save_server = config.get("save_dir", "save_server")
    logs_client = config.get("save_dir","logs_client")
    logs_server = config.get("save_dir", "logs_server")
    
    # Ger federation settings
    time_termination = int(config.get("federation", "time_termination"))
    server_port = int(config.get("federation", "server_port"))
        
    # Define 'client' options for GRPC communication (80*1024*1024)
    MAX_MESSAGE_LENGTH = int(config.get("grpc", "max_message_length")) 
    MAX_INBOUND_MESSAGE_SIZE = int(config.get("grpc", "max_inbound_message_size"))
    MAX_INBOUND_METADATA_SIZE = int(config.get("grpc", "max_inbound_metadata_size"))
    opts_client = [
        ('grpc.max_message_length', MAX_MESSAGE_LENGTH),
        ('grpc.max_send_message_length', MAX_MESSAGE_LENGTH),
        ('grpc.max_receive_message_length', MAX_MESSAGE_LENGTH),
        ('grpc.max_inbound_message_size', MAX_INBOUND_MESSAGE_SIZE),
        ('grpc.max_inbound_metadata_size', MAX_INBOUND_METADATA_SIZE),
        ('grpc.max_metadata_size', MAX_INBOUND_METADATA_SIZE)
    ]

    # Define 'server' options for GRPC communication
    opts_server = [
        ("grpc.keepalive_time_ms", int(config.get("grpc", "keepalive_time_ms"))),
        ("grpc.keepalive_timeout_ms", int(config.get("grpc", "keepalive_timeout_ms"))),
        ("grpc.keepalive_permit_without_calls", True if config.get("grpc", "keepalive_permit_without_calls") == "True" else False),
        ("grpc.http2.max_ping_strikes", int(config.get("grpc", "max_ping_strikes")))]

    if args.id == 0:
        print("Starting server with", args.min_clients_federation,
              "as minimum number of clients to start the federation.")
        start_server(
            min_num_clients=args.min_clients_federation,
            model_type=args.model_type,
            max_iters=args.max_iters,
            opts_server=opts_server,
            opts_client=opts_client,
            address=address,
            training_params=training_params,
            save_server=save_server,
            logs_server=logs_server,
            server_port=server_port,
            time_termination=time_termination)
    else:
        print("Starting client with id ", args.id)
        start_client(
            id_client=args.id,
            data_type=args.data_type,
            fos=args.fos,
            source=args.source,
            address=address,
            opts_client=opts_client,
            opts_server=opts_server,
            save_client=save_client,
            logs_client=logs_client)

if __name__ == "__main__":
    main()
