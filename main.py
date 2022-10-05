# -*- coding: utf-8 -*-
"""
Created on Feb 4, 2022

.. codeauthor:: L. Calvo-Bartolomé (lcalvo@pa.uc3m.es)
"""

import argparse
import json
import multiprocessing as mp
import pathlib
from subprocess import check_output
import sys
from concurrent import futures

import dask.dataframe as dd
import grpc
import numpy as np
import pandas as pd

from src.federation.client import Client
from src.federation.server import FederatedServer
from src.preprocessing.text_preproc import textPreproc
from src.protos import federated_pb2_grpc


do_spark = True

####################################################
################### TRAINING PARAMS ################
####################################################
# Training params
fixed_parameters = {
    "contextual_size": 10,
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
    "batch_size": 64,
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

# Training data
file_synthetic = "data/training_data/synthetic2.npz"  # workspace/
file_preproc_real = ""
path_real = "data/training_data"
####################################################

nw = 0


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


def start_client(id_client, data_type, fos):
    """Initialize a client that is going to contribute to the training of a federated topic model.

    Parameters
    ----------
    id_client : int
        Client's identifier
    data_type : str
        Type of the data that is going to be used by the client for the training (synthetic|real)
    """

    if data_type == "synthetic":
        data = np.load(file_synthetic, allow_pickle=True)
        corpus = data['documents'][id_client-1]
        vocab_size = data['vocab_size']
        word_topic_distrib_gt = data['topic_vectors']
        doc_topic_distrib_gt_all = data['doc_topics']
        doc_topic_distrib_gt_together = []
        for i in np.arange(len(doc_topic_distrib_gt_all)):
            doc_topic_distrib_gt_together.extend(doc_topic_distrib_gt_all[i])

    elif data_type == "real":
        # TODO: Revise if this works like that
        corpusFile = pathlib.Path(path_real).joinpath('corpus.parquet')
        if not corpusFile.is_dir():
            sys.exit(
                "The corpus file 'corpus.parquet' does not exist.")
        df = pd.read_parquet(corpusFile)
        df = df[df['fieldsOfStudy'] == fos]
        corpus = df
        print(df.columns)

    else:
        print("Specified data type not supported")

    # START CLIENT
    # Open channel for communication with the server
    MAX_MESSAGE_LENGTH = 20 * 1024 * 1024
    options = [
        ('grpc.max_message_length', MAX_MESSAGE_LENGTH),
        ('grpc.max_send_message_length', MAX_MESSAGE_LENGTH),
        ('grpc.max_receive_message_length', MAX_MESSAGE_LENGTH),
    ]
    # gfedntm-server
    with grpc.insecure_channel('localhost:50051', options=options) as channel:
        stub = federated_pb2_grpc.FederationStub(channel)

        # Create client
        client = Client(id=id_client,
                        stub=stub,
                        local_corpus=corpus,
                        data_type=data_type)

        # Start training of local model
        client.train_local_model()

        # Print evaluation results if data_type is synthetic
        if data_type == "synthetic":
            eval_parameters = [
                vocab_size, doc_topic_distrib_gt_all[id_client-1], word_topic_distrib_gt]
            client.eval_local_model(eval_params=eval_parameters)


def preproc():
    """Carries out simple preprocessing tasks and prepare a training file in the format required by the federation.
    """

    configFile = pathlib.Path("aux/config.json")
    script_spark = "/export/usuarios_ml4ds/lbartolome/spark/script-spark"
    token_spark = "/export/usuarios_ml4ds/lbartolome/spark/tokencluster.json"
    script_path = './src/preprocessing/text_preproc.py'
    machines = 10
    cores = 4
    options = '"--spark --preproc --config ' + configFile.resolve().as_posix() + '"'
    cmd = script_spark + ' -C ' + token_spark + \
        ' -c ' + cores + ' -N ' + machines + ' -S ' + script_path + ' -P ' + options
    print(cmd)
    try:
        print(f'-- -- Running command {cmd}')
        output = check_output(args=cmd, shell=True)
    except:
        print('-- -- Execution of script failed')


    # # Loads configuration file
    # configFile = pathlib.Path("aux/config.json")
    # if configFile.is_file():
    #     with configFile.open('r', encoding='utf8') as fin:
    #         train_config = json.load(fin)

    # tPreproc = textPreproc(stw_files=train_config['Preproc']['stopwords'],
    #                        eq_files=train_config['Preproc']['equivalences'],
    #                        min_lemas=train_config['Preproc']['min_lemas'],
    #                        no_below=train_config['Preproc']['no_below'],
    #                        no_above=train_config['Preproc']['no_above'],
    #                        keep_n=train_config['Preproc']['keep_n'])

    # # Create a Dataframe with all training data
   
    # trDtFile = pathlib.Path(train_config['TrDtSet'])
   
    # with trDtFile.open() as fin:
    #     trDtSet = json.load(fin)

    # if do_spark:
      
    #     # Read all training data and configure them as a spark dataframe
    #     for idx, DtSet in enumerate(trDtSet['Dtsets']):
    #         df = spark.read.parquet(f"file://{DtSet['parquet']}")
    #         if len(DtSet['filter']):
    #             # To be implemented
    #             # Needs a spark command to carry out the filtering
    #             # df = df.filter ...
    #             pass
    #         df = (
    #             df.withColumn("all_lemmas", F.concat_ws(
    #                 ' ', *DtSet['lemmasfld']))
    #                 .withColumn("source", F.lit(DtSet["source"]))
    #                 .select("id", "source", "all_lemmas")
    #         )
    #         if idx == 0:
    #             trDF = df
    #         else:
    #             trDF = trDF.union(df).distinct()

    #     # We preprocess the data and save the CountVectorizer Model used to obtain the BoW
    #     trDF = tPreproc.preprocBOW(trDF)
    #     tPreproc.saveCntVecModel(configFile.parent.resolve())

    #     # If the trainer is CTM, we also need the embeddings
    #     if train_config['trainer'] == "ctm":
    #         # We get full df containing the embeddings
    #         for idx, DtSet in enumerate(trDtSet['Dtsets']):
    #             df = spark.read.parquet(f"file://{DtSet['parquet']}")
    #             df = df.select("id", "embeddings", "fieldsOfStudy")
    #             if idx == 0:
    #                 eDF = df
    #             else:
    #                 eDF = eDF.union(df).distinct()
    #         # We perform a left join to keep the embeddings of only those documents kept after preprocessing
    #         # TODO: Check that this is done properly in Spark
    #         trDF = (trDF.join(eDF, trDF.id == eDF.id, "left")
    #                 .drop(df.id))

    #     trDataFile = tPreproc.exportTrData(trDF=trDF,
    #                                         dirpath=configFile.parent.resolve(),
    #                                         tmTrainer=train_config['trainer'])
    #     sys.stdout.write(trDataFile.as_posix())
    # else:

    #     # Read all training data and configure them as a dask dataframe
    #     for idx, DtSet in enumerate(trDtSet['Dtsets']):

    #         df = dd.read_parquet(DtSet['parquet']).fillna("")

    #         # Concatenate text fields
    #         for idx2, col in enumerate(DtSet['lemmasfld']):
    #             if idx2 == 0:
    #                 df["all_lemmas"] = df[col]
    #             else:
    #                 df["all_lemmas"] += " " + df[col]
    #         df["source"] = DtSet["source"]
    #         df = df[["id", "source", "all_lemmas"]]

    #         # Concatenate dataframes
    #         if idx == 0:
    #             trDF = df
    #         else:
    #             trDF = dd.concat([trDF, df])

    #     trDF = tPreproc.preprocBOW(trDF, nw)
    #     tPreproc.saveGensimDict(pathlib.Path(path_real))

    #     # We get full df containing the embeddings
    #     for idx, DtSet in enumerate(trDtSet['Dtsets']):
    #         df = dd.read_parquet(DtSet['parquet']).fillna("")
    #         df = df[["id", "embeddings", "fieldsOfStudy"]]

    #         # Concatenate dataframes
    #         if idx == 0:
    #             eDF = df
    #         else:
    #             eDF = dd.concat([trDF, df])

    #     # We perform a left join to keep the embeddings of only those documents kept after preprocessing
    #     trDF = trDF.merge(eDF, how="left", on=["id"])

    #     trDataFile = tPreproc.exportTrData(trDF=trDF,
    #                                     dirpath=pathlib.Path(path_real),
    #                                     tmTrainer="ctm",
    #                                     nw=nw)
    # print("Preprocessed file save: ", trDataFile.as_posix())


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
    parser.add_argument('--data_type', type=str, default="synthetic",
                        help="synthetic or real")
    parser.add_argument('--preproc', action='store_true', default=False,
                        help="Preprocess training data according to config file")
    # TODO: Add this in the calling to the server
    parser.add_argument("--fos", type=str, default="s2cs",
                        help="Category")
    args = parser.parse_args()

    if args.preproc:
        print("Carrying out preprocessing of ", file_preproc_real)
        preproc()
    elif args.id == 0:
        print("Starting server with", args.min_clients_federation,
              "as minimum number of clients to start the federation.")
        start_server(args.min_clients_federation, args.model_type)
    else:
        print("Starting client with id ", args.id)
        start_client(args.id, args.data_type, args.fos)


if __name__ == "__main__":
    main()
