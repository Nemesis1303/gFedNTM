# **gFedNTM**

[![arXiv](https://img.shields.io/badge/arXiv-2212.02269-<COLOR>.svg)](https://arxiv.org/abs/2212.02269)
[![PyPI - License](https://img.shields.io/badge/license-MIT-green.svg)](https://github.com/Nemesis1303/gFedNTM/blob/main/LICENSE)

``gFedNTM`` is a general federated framework for training neural topic models (NTMs) that utilizes Google Remote Procedure Call (gRPC) for server-client communication.

## Table of contents

- [**gFedNTM**](#gfedntm)
  - [Table of contents](#table-of-contents)
  - [Overview](#overview)
  - [Available NTMs](#available-ntms)
  - [Usage](#usage)
  - [Preprocessing](#preprocessing)
  - [Dataset format](#dataset-format)
    - [Synthetic datasets](#synthetic-datasets)
    - [Real datasets](#real-datasets)
  - [Code Structure](#code-structure)
  - [Generate gRPC Python files from proto file (from ``src/protos``)](#generate-grpc-python-files-from-proto-file-from-srcprotos)
  - [License](#license)

## Overview

In our method, training takes place in two differentiated sequential stages (illustrated below):

1. **Vocabulary consensus**. The server waits until the vocabulary of all nodes has been received and then merges them into one common one used to initialize the global model.

2. **Federated training**.  It begins after the clients have received back the latter two from the server. Then, at each mini-batch step, the server waits for all the clients to send their gradients, aggregates them, and sends the updated global model parameters back to the clients. This process is repeated until some convergence criterion is met.

    ![gFedNTM's training workflow](https://github.com/Nemesis1303/gFedNTM/blob/main/static/images/federated_diag.png?raw=true)

## Available NTMs

It currently supports **implementations for three existing state-of-the-art NTMs:**

|                     Name                     |                         Source code                         |
| :------------------------------------------: | :---------------------------------------------------------: |
|    **CTMs** `(Bianchi et al. 2020, 2021)`    | <https://github.com/MilaNLProc/contextualized-topic-models> |
| **NeuralLDA** `(Srivastava and Sutton 2017)` |       <https://github.com/estebandito22/PyTorchAVITM>       |
|  **ProdLDA** `(Srivastava and Sutton 2017)`  |       <https://github.com/estebandito22/PyTorchAVITM>       |

## Usage

This project has been prepared to run through **Docker-compose**, where each node (i.e., client) or server is considered a "service". Alternatively, you can start each service independently.

The [``docker-compose``](https://github.com/Nemesis1303/gFedNTM/blob/main/docker-compose.yaml) file allows you to easily spin up multiple instances of the gFedNTM server and clients.

For this, it is first necessary to:

1. If desired, configure the topic modeling hyperparameters, as well as some settings for the federation (waiting time, server port, etc.) by modifying the settings available at the [config file](https://github.com/Nemesis1303/gFedNTM/blob/main/config/dft_params.cf).

2. Build the [``Dockerfile``](https://github.com/Nemesis1303/gFedNTM/blob/main/Dockerfile) image :

    ```bash
    docker build .  -t nemesis1303/gfedntm:latest
    ```

3. Modify the ``docker-compose`` to define your needs. In any case, it should always contain **one server** and **as many clients as you desire to have in your federation**. By default, the docker-compose includes the following services:

   - ``gfedntm-dev:`` This service is used for development purposes and is not required to run the server or clients.
   - ``gfedntm-server:`` This service runs the gFedNTM server and exposes it on port 8888. The command section allows you to pass in arguments to the ``main.py`` script to configure the server, as defined below:

      ```bash
      command: python3 workspace/main.py --min_clients_federation <min_num_clients> --model_type <model_type> --max_iters <max_iters>
      ```

      where:

      - ``<min_num_clients>`` is the minimum number of clients required for the federation to begin
      - ``<model_type>`` is the underlying topic modeling algorithm with which the federated topic model is constructed.
      - ``<model_tymax_iterspe>`` is the maximum number of global iterations to train the federated topic model.

   - ``gfedntm-client1 to gfedntm-client5:`` These services run the gFedNTM clients and expose them on different ports. The command section allows you to pass in arguments to the ``main.py`` script to set up the clients, as defined below:

      ```bash
      command: python3 workspace/main.py start_client --id <id_client> --source <sourceFile> --data_type <data_type> --fos <fos>
      ```

      where:

      - ``<id_client>`` is the client's identifier (it must start from 1 on since 0 is the identifier of the server by default)
      - ``<source>`` is the path to the data that is going to be used by the client for the training
      - ``<data_type>`` is the type of data that will be used by the client for training (synthetic or real).
      - ``<fos>`` is the category or label describing the data in the source belonging to each client.
  
    All client services are mounted to the ``./static/output_models/client<id_client>`` and ``./static/logs/client<id_client>`` directories, where the resulting trained models and logs generated during the process are saved. For the server the paths to these directories are ``./static/output_models/server`` and ``./static/logs/server``, respectively.

## Preprocessing

Preparing your training files in the required format for the library is easy with the help of the [``text_preproc.py``](https://github.com/Nemesis1303/gFedNTM/blob/main/aux_scripts/preprocessing/text_preproc.py) script. This script handles simple preprocessing tasks, ensuring your data is ready for training.

To start the preprocessing, you have two options:

- Utilize Spark: If you choose this route, make sure you have a Spark cluster available.
- Use Gensim with Dask acceleration: This option provides efficient preprocessing.

The script leverages the [topicmodeler](https://github.com/IntelCompH2020/topicmodeler) GitHub project., streamlining the preprocessing process. Please note that your familiarity with this project might impact your preprocessing choices.

## Dataset format

The code gives support to two different types of datasets, what we consider ``real`` and ``synthetic`` datasets.

### Synthetic datasets

A synthetic dataset is a collection of documents that have been generated using LDA's generative model. However, unlike real documents, the words in the synthetic dataset do not have any semantic meaning and are labeled with terms like "term56" and "term125". The main advantage of this type of dataset is that it allows us to compare the performance of a federated model against individual models generated by each node. This is possible because we have the actual document-topic and topic-word distributions of the documents. For more information, please refer to [our paper](https://arxiv.org/pdf/2212.02269.pdf).

If you would like to generate a synthetic dataset, you can use the script available at ``src/utils/generate_synthetic.py``. An example of a synthetic corpus for ``N=5`` nodes, with ``1000`` documents for each node, is also available at ``static/datasets/synthetic.npz``.

### Real datasets

Real datasets are sets of documents that are collected from actual real-world situations, which are used to evaluate the performance of federated models in real-world scenarios. Using real datasets provides an advantage in that they offer a more accurate representation of the data that a model will encounter in the real world. However, it can be difficult to obtain real datasets due to privacy issues and data access constraints. Furthermore, real datasets may require preprocessing or cleaning before they can be effectively used for training or evaluation.

For the code to function properly, the dataset must be provided in parquet format and contain the following columns:

- ``bow_text:``This column contains the actual text that will be analyzed, which are the lemmas of the text fields chosen for analysis in the original dataset (e.g., title and abstract).
- ``embeddings:``It is only required when using the CTM algorithm and contains the embeddings of each of the documents under analysis. Take into account that if you use CTM as the underlying algorithm, you will need to modify the parameter ``contextual_size`` in ``fixed_parameters`` in the ``main.py`` according to the size your embeddings have.
- ``fos:``This column is a label or category that identifies the documents of each model.

An example of how a training file should look is provided at ``static/datasets/preprocessing_example/s2cs_tiny_preproc.parquet``, which contains a preprocessed small sample of the Semantic Scholar dataset in the categories of Computer Science, Economics, Sociology, Philosophy, and Political Science. Each category's documents (the concatenation of the articles' textual fields) are assigned to a different client.

## Code Structure

The repository is organized as follows:

```bash
gFedNTM/
├── docker-compose.yaml
├── Dockerfile
├── config/
├── experiments_centralized/
│   ├── dft_params.cf
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
│       ├── generate_synthetic.py
│       ├── utils_postprocessing.py
│       └── utils_preprocessing.py
└── static/
```

> *Please note that only the most important parts of the code have been described in detail in this directory structure overview.*

## Generate gRPC Python files from proto file (from ``src/protos``)

To generate the python proto files you must run the following command:

```python
python3 -m grpc_tools.protoc -I . --python_out=. --grpc_python_out=. ./federated.proto
```

The generated file ``federated_pb2.py`` contains the type definitions, while ``federated_pb2_grpc.py`` describes the framework for a client and a server.

> *Note that the execution of this command is only necessary if the ``federated.proto`` file is modified.*

## License

This project is licensed under the MIT License - see the `LICENSE` file for details.
