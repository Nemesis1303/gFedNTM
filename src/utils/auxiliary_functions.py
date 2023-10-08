# -*- coding: utf-8 -*-
"""
Created on Feb 1, 2022
Last updated on Aug 24, 2023

.. codeauthor:: L. Calvo-Bartolomé (lcalvo@pa.uc3m.es)
"""
import configparser
import pickle
import torch
from scipy import sparse
from typing import Union
import numpy as np
from scipy import sparse
from sklearn.preprocessing import normalize
from src.protos import federated_pb2

##############################################################################
#                                CONSTANTS                                   #
##############################################################################
CHUNK_SIZE = 1024 * 1024  # 1MB


def get_type_from_string(str_dtype):
    """Gets the dtype object from its string characterization"""
    if str_dtype == "float32":
        dtype = np.float32
    elif str_dtype == "float64":
        dtype = np.float64
    elif str_dtype == "int64":
        dtype = np.int64
    else:
        print("dtype not defined")
        print(str_dtype)
    return dtype


def get_file_chunks(filename):
    with open(filename, 'rb') as f:
        while True:
            piece = f.read(CHUNK_SIZE)
            if len(piece) == 0:
                return
            yield federated_pb2.Chunk(buffer=piece)


def save_chunks_to_file(chunks, filename):
    with open(filename, 'wb') as f:
        for chunk in chunks:
            f.write(chunk.buffer)


def unpickler(file: str):
    """Unpickle file"""
    with open(file, 'rb') as f:
        return pickle.load(f)


def pickler(file: str, ob):
    """Pickle object to file"""
    with open(file, 'wb') as f:
        pickle.dump(ob, f)
    return


def save_model_as_npz(
    npzfile,
    federated_model,
) -> None:
    """Saves the matrixes that characterize a topic model in a numpy npz filel.

    Parameters
    ----------
    npzfile: str
        Name of the file in which the model will be saved
    federated_model: Union[FederatedCTM,FederatedAVITM]
        Federated model to be saved
    """

    if isinstance(federated_model.thetas, sparse.csr_matrix):
        np.savez(
            npzfile,
            betas=federated_model.betas,
            thetas_data=federated_model.thetas.data,
            thetas_indices=federated_model.thetas.indices,
            thetas_indptr=federated_model.thetas.indptr,
            thetas_shape=federated_model.thetas.shape,
            ntopics=federated_model.n_components,
            topics=federated_model.topics
        )
    else:
        np.savez(
            npzfile,
            betas=federated_model.betas,
            thetas=federated_model.thetas,
            ntopics=federated_model.n_components,
            topics=federated_model.topics
        )
    return


def serializeTensor(tensor: torch.Tensor) -> federated_pb2.Tensor:
    """Serializes a tensor into a protobuf message.

    Parameters
    ----------
    tensor: torch.Tensor
        Tensor to be serialized.

    Returns
    -------
    serializedTensor: federated_pb2.Tensor
        Protobuf message containing the serialized tensor.
    """

    aux = tensor.cpu().numpy()
    content_bytes = aux.tobytes()
    content_type = str(aux.dtype)
    size = federated_pb2.TensorShape()
    num_dims = len(tensor.shape)
    for i in np.arange(num_dims):
        name = "dim" + str(i)
        size.dim.extend(
            [federated_pb2.TensorShape.Dim(size=tensor.shape[i], name=name)])
    serializedTensor = federated_pb2.Tensor(tensor_shape=size,
                                            dtype=content_type,
                                            tensor_content=content_bytes)
    return serializedTensor


def deserializeTensor(protoTensor: federated_pb2.Tensor) -> torch.Tensor:
    """Deserializes a protobuf message into a tensor.

    Parameters
    ----------
    protoTensor: federated_pb2.Tensor
        Protobuf message containing the serialized tensor.

    Returns
    -------
    deserialized_tensor: torch.Tensor
        Tensor deserialized from the protobuf message.
    """

    deserialized_numpy = deserializeNumpy(protoTensor)
    deserialized_tensor = torch.Tensor(deserialized_numpy)

    return deserialized_tensor


def deserializeNumpy(protoTensor: federated_pb2.Tensor) -> np.ndarray:
    """Deserializes a protobuf message into a numpy array.

    Parameters
    ----------
    protoTensor: federated_pb2.Tensor
        Protobuf message containing the serialized tensor.

    Returns
    -------
    deserialized_numpy: np.ndarray
        Numpy array deserialized from the protobuf message.
    """

    dims = tuple(
        [dim.size for dim in protoTensor.tensor_shape.dim])
    dtype_send = get_type_from_string(protoTensor.dtype)
    deserialized_bytes = np.frombuffer(
        protoTensor.tensor_content, dtype=dtype_send)
    deserialized_numpy = np.reshape(
        deserialized_bytes, newshape=dims)

    return deserialized_numpy


def optStateDict_to_proto(optStateDict: dict) -> federated_pb2.OptUpdate:
    """Converts a dictionary containing the optimizer state into a protobuf message.

    Parameters
    ----------
    optStateDict: dict
        Dictionary containing the optimizer state.

    Returns
    -------
    optUpdate: federated_pb2.OptUpdate
        Protobuf message containing the optimizer state.
    """

    for i in optStateDict.keys():
        if i == "state":
            # We iterate first over the state dictionary
            state_ = federated_pb2.AdamUpdate.State()
            for j in optStateDict[i].keys():  # 0
                # Read step
                step_ = serializeTensor(optStateDict[i][j]["step"])
                # Read exp_avg
                exp_avg_ = serializeTensor(optStateDict[i][j]["exp_avg"])
                # Read exp_avg_sq
                exp_avg_sq_ = serializeTensor(optStateDict[i][j]["exp_avg_sq"])
                contentState = \
                    federated_pb2.AdamUpdate.State.ContentState(
                        state_id=j,
                        step=step_,
                        exp_avg=exp_avg_,
                        exp_avg_sq=exp_avg_sq_
                    )
                state_.contentState.extend([contentState])

        elif i == "param_groups":
            # We iterate over the param_groups list
            for dic in optStateDict[i]:
                for key in dic:
                    if key == "lr":
                        lr_ = dic[key]
                    elif key == "betas":
                        aux = list(dic[key])
                        betas_ = federated_pb2.AdamUpdate.ParamGroups.Betas(
                            beta1=aux[0],
                            beta2=aux[1]
                        )
                    elif key == "eps":
                        eps_ = dic[key]
                    elif key == "weight_decay":
                        weight_decay_ = dic[key]
                    elif key == "amsgrad":
                        amsgrad_ = dic[key]
                    elif key == "params":
                        params_ = dic[key]
                    elif key == "maximize":
                        params_ = dic[key]
                    else:
                        print("Wrong key found when protofying param_groups")
                        print(key)

            if None in [lr_, betas_, eps_, weight_decay_, amsgrad_, params_]:
                print("Something went wrong protofying param_groups")
                return

            paramGroups_ = federated_pb2.AdamUpdate.ParamGroups(
                lr=lr_,
                betas=betas_,
                eps=eps_,
                weight_decay=weight_decay_,
                amsgrad=amsgrad_
            )
            for param in params_:
                paramGroups_.params.extend([param])

    adamUpdate_ = federated_pb2.AdamUpdate(
        state=state_,
        paramGroups=paramGroups_
    )

    return federated_pb2.OptUpdate(adamUpdate=adamUpdate_)


def proto_to_optStateDict(optUpdate: federated_pb2.OptUpdate) -> dict:
    """Converts a protobuf message containing the optimizer state into a dictionary.

    NOTE: Currently only works for Adam optimizer.

    Parameters
    ----------
    optUpdate: federated_pb2.OptUpdate
        Protobuf message containing the optimizer state.

    Returns
    -------
    optStateDict: dict
        Dictionary containing the optimizer state.
    """

    state = optUpdate.adamUpdate.state
    paramGroups = optUpdate.adamUpdate.paramGroups

    stateDict = {}
    for cs in state.contentState:
        stateDict[cs.state_id] = {
            "step": cs.step,
            "exp_avg": deserializeTensor(cs.exp_avg),
            "exp_avg_sq": deserializeTensor(cs.exp_avg_sq)
        }
    paramGroupsList = [
        {"lr": paramGroups.lr,
         "betas": tuple([paramGroups.betas.beta1, paramGroups.betas.beta2]),
         "eps": paramGroups.eps,
         "weight_decay": paramGroups.weight_decay,
         "amsgrad": paramGroups.amsgrad,
         "params": [param for param in paramGroups.params]}
    ]

    optStateDict = {
        "state": stateDict,
        "param_groups": paramGroupsList
    }

    return optStateDict


def modelStateDict_to_proto(
    modelStateDict: dict,
    current_epoch: int,
    model_type
) -> federated_pb2.ModelUpdate:
    """Transforms a dictionary containing the model state of a certain epoch into a protobuf message.

    Parameters
    ----------
    modelStateDict: dict
        Dictionary containing the model state.
    current_epoch: int
        Current epoch of the model.

    Returns
    -------
    modelUpdate: federated_pb2.ModelUpdate
        Protobuf message containing the model state.
    """

    new_dict = {}
    new_dict["current_epoch"] = current_epoch
    for key in list(modelStateDict.keys()):
        if key == 'inf_net.hiddens.l_0.0.weight':
            new_key = "inf_net_hiddens_l00_weight"
        elif key == 'inf_net.hiddens.l_0.0.bias':
            new_key = "inf_net_hiddens_l_00_bias"
        else:
            new_key = key.replace(".", "_")
        new_dict[new_key] = serializeTensor(
            modelStateDict[key])

    return federated_pb2.ModelUpdate(**new_dict)


def proto_to_modelStateDict(modelUpdate: federated_pb2.ModelUpdate) -> dict:
    """Transforms a protobuf message containing the model state into a dictionary.

    Parameters
    ----------
    modelUpdate: federated_pb2.ModelUpdate
        Protobuf message containing the model state.

    Returns
    -------
    modelStateDict: dict
        Dictionary containing the model state.
    """

    field_mappings = {
        "inf_net_f_mu_batchnorm_num_batches_tracked": "inf_net.f_mu_batchnorm.num_batches_tracked",
        "inf_net_f_sigma_batchnorm_num_batches_tracked": "inf_net.f_sigma_batchnorm.num_batches_tracked",
        "beta_batchnorm_num_batches_tracked": "beta_batchnorm.num_batches_tracked",
        "prior_mean": "prior_mean",
        "prior_variance": "prior_variance",
        "beta": "beta",
        "inf_net_input_layer_weight": "inf_net.input_layer.weight",
        "inf_net_input_layer_bias": "inf_net.input_layer.bias",
        "inf_net_hiddens_l00_weight": "inf_net.hiddens.l_0.0.weight",
        "inf_net_hiddens_l_00_bias": "inf_net.hiddens.l_0.0.bias",
        "inf_net_f_mu_weight": "inf_net.f_mu.weight",
        "inf_net_f_mu_bias": "inf_net.f_mu.bias",
        "inf_net_f_mu_batchnorm_running_mean": "inf_net.f_mu_batchnorm.running_mean",
        "inf_net_f_mu_batchnorm_running_var": "inf_net.f_mu_batchnorm.running_var",
        "inf_net_f_sigma_weight": "inf_net.f_sigma.weight",
        "inf_net_f_sigma_bias": "inf_net.f_sigma.bias",
        "inf_net_f_sigma_batchnorm_running_mean": "inf_net.f_sigma_batchnorm.running_mean",
        "inf_net_f_sigma_batchnorm_running_var": "inf_net.f_sigma_batchnorm.running_var",
        "beta_batchnorm_running_mean": "beta_batchnorm.running_mean",
        "beta_batchnorm_running_var": "beta_batchnorm.running_var",
        "topic_word_matrix": "topic_word_matrix",
        "inf_net_adapt_bert_weight": "inf_net.adapt_bert.weight",
        "inf_net_adapt_bert_bias": "inf_net.adapt_bert.bias",
    }

    modelStateDict = {}

    for field_name, key_name in field_mappings.items():
        if modelUpdate.HasField(field_name):
            tensor = deserializeTensor(getattr(modelUpdate, field_name))
            if tensor.tolist() == []:
                tensor = torch.tensor(0, dtype=torch.long)
            modelStateDict[key_name] = tensor

    return modelStateDict

def read_config_experiments(file_path, skip=[]):
    """Reads the configuration file of the experiments.

    Parameters
    ----------
    file_path: str
        Path to the configuration file.
    skip: list
        List of sections to skip.

    Returns
    -------
    config_dict: dict
        Dictionary containing the configuration parameters.
    """
    config = configparser.ConfigParser()
    config.read(file_path)

    # Initialize an empty dictionary
    config_dict = {}

    # Loop through each section in the configuration file
    for section in config.sections():
        if section not in skip:
            # Retrieve the options and values within each section
            options = config.options(section)

            # Loop through each option in the section
            for option in options:
                # Retrieve the value of each option
                value = config.get(section, option)
                # Store the option-value pair in the section dictionary
                if option in ['n_components', 'num_iterations', 'batch_size', 'num_threads', 'optimize_interval', 'num_epochs', 'num_samples', 'num_data_loader_workers', 'contextual_size']:

                    config_dict[option] = int(value)
                elif option in ['thetas_thr', 'doc_topic_thr',
                                'alpha', 'dropout', 'lr',
                                'momentum', 'topic_prior_mean']:
                    config_dict[option] = float(value)
                elif option == "labels":
                    config_dict[option] = ""
                elif option == "topic_prior_variance":
                    config_dict[option] = None
                elif option in ["learn_priors", "reduce_on_plateau", "verbose"]:
                    config_dict[option] = True if value == "True" else False
                elif option == "hidden_sizes":
                    config_dict[option] = tuple(
                        map(int, value[1:-1].split(',')))
                else:
                    config_dict[option] = value

    return config_dict


def convert_topic_word_to_init_size(vocab_size: int,
                                    model,
                                    model_type: str,
                                    ntopics: int,
                                    id2token: list[tuple],
                                    all_words: list[str]):
    """It converts the topic-word distribution matrix obtained from the training of a model into a matrix with the dimensions of the original topic-word distribution, assigning zeros to those words that are not present in the corpus. 
    It is only of use in case we are training a model over a synthetic dataset, so as to later compare the performance of the attained model in what regards to the similarity between the original and the trained model.

    Parameters
    ----------
    vocab_size: int
        Size of the synethic'data vocabulary
    model: 
        Model whose topic-word matrix is being transformed
    model_type: str
        Type of the trained model (e.g. AVITM)
    ntopics: int
        Number of topics of the trained model
    id2token: List[tuple]
        Mappings with the content of the document-term matrix
    all_words: List[str]
        List of all the words of the vocabulary of size vocab_size

    Returns
    -------
    np.ndarray: Topic-word distribution matrix of the trained model with the dimensions of the original topic-word distribution
    """

    if model_type == "avitm":
        w_t_distrib = np.zeros((ntopics, vocab_size), dtype=np.float64)
        wd = model.get_topic_word_distribution()
        for i in np.arange(ntopics):
            for idx, word in id2token.items():
                for j in np.arange(len(all_words)):
                    if all_words[j] == word:  # word.split("__")[1]: # word
                        w_t_distrib[i, j] = wd[i][idx]
                        break
        normalized_array = normalize(w_t_distrib, axis=1, norm='l1')
        return normalized_array
    else:
        print("Method not impleemnted for the selected model type")
        return None
