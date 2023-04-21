# **gFedNTM**

``gFedNTM`` is general federated framework for the training of neural topic models (NTMs) that utilizes Google Remote Procedure Call
(gRPC) for the server-client communication. 


![](https://github.com/Nemesis1303/gFedNTM/static/images/federated_diag.png?raw=true)


It currently supports **implementations for three existing state-of-the-art NTMs:**

| Name | Implementation |
|:---:|:---:|
| CTM `(Bianchi et al. 2021)` | https://github.com/MilaNLProc/contextualized-topic-models |
| NeuralLDA `(Srivastava and Sutton 2017)` | https://github.com/estebandito22/PyTorchAVITM |
| ProdLda `(Srivastava and Sutton 2017)` | https://github.com/estebandito22/PyTorchAVITM |



## Code Structure

The repository is organized as follows: 

```bash
gFedNTM/
├── docker-compose.yaml
├── Dockerfile
├── experiments_centralized/
├── LICENSE
├── main.py
├── notebooks/
├── README.md
├── requirements.txt
├── run_simulation.py
├── src/
│   ├── __init__.py
│   ├── federation/
│   │   ├── __init__.py
│   │   ├── client.py
│   │   ├── federated_trainer_manager.py
│   │   ├── federation.py
│   │   ├── federation_client.py
│   │   └── server.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── base/
│   │   │   ├── __init__.py
│   │   │   ├── contextualized_topic_models/
│   │   │   ├── pytorchavitm/
│   │   │   └── utils/
│   │   │       └── early_stopping/
│   │   └── federated/
│   │       ├── __init__.py
│   │       ├── federated_avitm.py
│   │       ├── federated_ctm.py
│   │       └── federated_model.py
│   ├── preprocessing/
│   │   │── __init__.py
│   │   └── text_preproc.py
│   ├── protos/
│   │   ├── __init__.py
│   │   ├── federated.proto
│   │   ├── federated_pb2.py
│   │   └── federated_pb2_grpc.py
│   └── utils/
│       ├── __init__.py
│       ├── auxiliary_functions.py
│       ├── utils_postprocessing.py
│       └── utils_preprocessing.py
└── static/
```

> *Please note that only the most important parts of the code have been described in detail in this directory structure overview.*

## Usage

To use this project, follow these steps:

## Generate gRPC Python files from proto file (from ``src/protos``)

```
python3 -m grpc_tools.protoc -I . --python_out=. --grpc_python_out=. ./federated.proto
```

The generated file ``federated_pb2.py`` contains the type definitions, while ``federated_pb2_grpc.py`` describes the framework for a client and a server.

Note that the execution of this command is only necessary if the ``federated.proto`` file is modified.
## Run server

```
python3 main.py --min_clients_federation <min_clients_federation> --model_type <underlying_model_type> 
```

## Run client

```
python3 main.py --id <id> --source <path_to_training_data> 
```

## Build image

```
docker build .  -t nemesis1303/gfedntm:latest
```

## License

This project is licensed under the MIT License - see the `LICENSE` file for details.
