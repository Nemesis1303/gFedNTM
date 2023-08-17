# -*- coding: utf-8 -*-
"""
Created on Feb 1, 2022

@author: L. Calvo-Bartolomé (lcalvo@pa.uc3m.es)
"""
import configparser
import pickle

import numpy as np
import torch
from scipy import sparse

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


def save_model_as_npz(npzfile, federated_model):
    """Saves the matrixes that characterize a topic model in a numpy npz filel.

    Args:
        npzfile (str): Name of the file in which the model will be saved
        client (): 
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


def serializeTensor(tensor) -> federated_pb2.Tensor:

    if tensor.requires_grad == True:
        tensor = tensor.detatch()

    content_bytes = tensor.numpy().tobytes()
    content_type = str(tensor.numpy().dtype)
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

    deserialized_numpy = deserializeNumpy(protoTensor)
    deserialized_tensor = torch.Tensor(deserialized_numpy)

    return deserialized_tensor


def deserializeNumpy(protoTensor: federated_pb2.Tensor) -> np.ndarray:

    dims = tuple(
        [dim.size for dim in protoTensor.tensor_shape.dim])
    dtype_send = get_type_from_string(protoTensor.dtype)
    deserialized_bytes = np.frombuffer(
        protoTensor.tensor_content, dtype=dtype_send)
    deserialized_numpy = np.reshape(
        deserialized_bytes, newshape=dims)

    return deserialized_numpy


def optStateDict_to_proto(optStateDict) -> federated_pb2.OptUpdate:

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


def proto_to_optStateDict(modelUpdate: federated_pb2.OptUpdate) -> dict:

    # TODO: Update for more optimizers
    state = modelUpdate.adamUpdate.state
    paramGroups = modelUpdate.adamUpdate.paramGroups

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


def modelStateDict_to_proto(modelStateDict, current_epoch, model_type) -> federated_pb2.ModelUpdate:

    if model_type == "ctm":
        modelUpdate = federated_pb2.ModelUpdate(
            prior_mean=serializeTensor(modelStateDict['prior_mean']),
            prior_variance=serializeTensor(modelStateDict['prior_variance']),
            beta=serializeTensor(modelStateDict['beta']),
            inf_net_input_layer_weight=serializeTensor(
                modelStateDict['inf_net.input_layer.weight']),
            inf_net_input_layer_bias=serializeTensor(
                modelStateDict['inf_net.input_layer.bias']),
            inf_net_hiddens_l00_weight=serializeTensor(
                modelStateDict['inf_net.hiddens.l_0.0.weight']),
            inf_net_hiddens_l_00_bias=serializeTensor(
                modelStateDict['inf_net.hiddens.l_0.0.bias']),
            inf_net_f_mu_weight=serializeTensor(
                modelStateDict['inf_net.f_mu.weight']),
            inf_net_f_mu_bias=serializeTensor(modelStateDict['inf_net.f_mu.bias']),
            inf_net_f_mu_batchnorm_running_mean=serializeTensor(
                modelStateDict['inf_net.f_mu_batchnorm.running_mean']),
            inf_net_f_mu_batchnorm_running_var=serializeTensor(
                modelStateDict['inf_net.f_mu_batchnorm.running_var']),
            inf_net_f_mu_batchnorm_num_batches_tracked=serializeTensor(
                modelStateDict['inf_net.f_mu_batchnorm.num_batches_tracked']),
            inf_net_f_sigma_weight=serializeTensor(
                modelStateDict['inf_net.f_sigma.weight']),
            inf_net_f_sigma_bias=serializeTensor(
                modelStateDict['inf_net.f_sigma.bias']),
            inf_net_f_sigma_batchnorm_running_mean=serializeTensor(
                modelStateDict['inf_net.f_sigma_batchnorm.running_mean']),
            inf_net_f_sigma_batchnorm_running_var=serializeTensor(
                modelStateDict['inf_net.f_sigma_batchnorm.running_var']),
            inf_net_f_sigma_batchnorm_num_batches_tracked=serializeTensor(
                modelStateDict['inf_net.f_sigma_batchnorm.num_batches_tracked']),
            beta_batchnorm_running_mean=serializeTensor(
                modelStateDict['beta_batchnorm.running_mean']),
            beta_batchnorm_running_var=serializeTensor(
                modelStateDict['beta_batchnorm.running_var']),
            beta_batchnorm_num_batches_tracked=serializeTensor(
                modelStateDict['beta_batchnorm.num_batches_tracked']),
            current_epoch=current_epoch,
            inf_net_adapt_bert_weight=serializeTensor(
                    modelStateDict['inf_net.adapt_bert.weight']),
            inf_net_adapt_bert_bias= serializeTensor(
                    modelStateDict['inf_net.adapt_bert.bias'])
        )
    else:
            modelUpdate = federated_pb2.ModelUpdate(
        prior_mean=serializeTensor(modelStateDict['prior_mean']),
        prior_variance=serializeTensor(modelStateDict['prior_variance']),
        beta=serializeTensor(modelStateDict['beta']),
        inf_net_input_layer_weight=serializeTensor(
            modelStateDict['inf_net.input_layer.weight']),
        inf_net_input_layer_bias=serializeTensor(
            modelStateDict['inf_net.input_layer.bias']),
        inf_net_hiddens_l00_weight=serializeTensor(
            modelStateDict['inf_net.hiddens.l_0.0.weight']),
        inf_net_hiddens_l_00_bias=serializeTensor(
            modelStateDict['inf_net.hiddens.l_0.0.bias']),
        inf_net_f_mu_weight=serializeTensor(
            modelStateDict['inf_net.f_mu.weight']),
        inf_net_f_mu_bias=serializeTensor(modelStateDict['inf_net.f_mu.bias']),
        inf_net_f_mu_batchnorm_running_mean=serializeTensor(
            modelStateDict['inf_net.f_mu_batchnorm.running_mean']),
        inf_net_f_mu_batchnorm_running_var=serializeTensor(
            modelStateDict['inf_net.f_mu_batchnorm.running_var']),
        inf_net_f_mu_batchnorm_num_batches_tracked=serializeTensor(
            modelStateDict['inf_net.f_mu_batchnorm.num_batches_tracked']),
        inf_net_f_sigma_weight=serializeTensor(
            modelStateDict['inf_net.f_sigma.weight']),
        inf_net_f_sigma_bias=serializeTensor(
            modelStateDict['inf_net.f_sigma.bias']),
        inf_net_f_sigma_batchnorm_running_mean=serializeTensor(
            modelStateDict['inf_net.f_sigma_batchnorm.running_mean']),
        inf_net_f_sigma_batchnorm_running_var=serializeTensor(
            modelStateDict['inf_net.f_sigma_batchnorm.running_var']),
        inf_net_f_sigma_batchnorm_num_batches_tracked=serializeTensor(
            modelStateDict['inf_net.f_sigma_batchnorm.num_batches_tracked']),
        beta_batchnorm_running_mean=serializeTensor(
            modelStateDict['beta_batchnorm.running_mean']),
        beta_batchnorm_running_var=serializeTensor(
            modelStateDict['beta_batchnorm.running_var']),
        beta_batchnorm_num_batches_tracked=serializeTensor(
            modelStateDict['beta_batchnorm.num_batches_tracked']),
        current_epoch=current_epoch,
    )
        
    return modelUpdate


def proto_to_modelStateDict(modelUpdate: federated_pb2.ModelUpdate) -> dict:

    aux = modelUpdate
    inf_net_f_mu_batchnorm_num_batches_tracked = \
        deserializeTensor(
            modelUpdate.inf_net_f_mu_batchnorm_num_batches_tracked)
    if inf_net_f_mu_batchnorm_num_batches_tracked.tolist() == []:
        inf_net_f_mu_batchnorm_num_batches_tracked = torch.tensor(
            0, dtype=torch.long)

    inf_net_f_sigma_batchnorm_num_batches_tracked = \
        deserializeTensor(
            modelUpdate.inf_net_f_sigma_batchnorm_num_batches_tracked)
    if inf_net_f_sigma_batchnorm_num_batches_tracked.tolist() == []:
        inf_net_f_sigma_batchnorm_num_batches_tracked = torch.tensor(
            0, dtype=torch.long)

    beta_batchnorm_num_batches_tracked = \
        deserializeTensor(modelUpdate.beta_batchnorm_num_batches_tracked)
    if beta_batchnorm_num_batches_tracked.tolist() == []:
        beta_batchnorm_num_batches_tracked = torch.tensor(0, dtype=torch.long)

    modelStateDict = {
        "prior_mean": deserializeTensor(modelUpdate.prior_mean),
        "prior_variance": deserializeTensor(modelUpdate.prior_variance),
        "beta":  deserializeTensor(modelUpdate.beta),
        "inf_net.input_layer.weight": deserializeTensor(modelUpdate.inf_net_input_layer_weight),
        "inf_net.input_layer.bias": deserializeTensor(modelUpdate.inf_net_input_layer_bias),
        "inf_net.hiddens.l_0.0.weight": deserializeTensor(modelUpdate.inf_net_hiddens_l00_weight),
        "inf_net.hiddens.l_0.0.bias": deserializeTensor(modelUpdate.inf_net_hiddens_l_00_bias),
        "inf_net.f_mu.weight": deserializeTensor(modelUpdate.inf_net_f_mu_weight),
        "inf_net.f_mu.bias": deserializeTensor(modelUpdate.inf_net_f_mu_bias),
        "inf_net.f_mu_batchnorm.running_mean": deserializeTensor(modelUpdate.inf_net_f_mu_batchnorm_running_mean),
        "inf_net.f_mu_batchnorm.running_var": deserializeTensor(modelUpdate.inf_net_f_mu_batchnorm_running_var),
        "inf_net.f_mu_batchnorm.num_batches_tracked": inf_net_f_mu_batchnorm_num_batches_tracked,
        "inf_net.f_sigma.weight": deserializeTensor(modelUpdate.inf_net_f_sigma_weight),
        "inf_net.f_sigma.bias": deserializeTensor(modelUpdate.inf_net_f_sigma_bias),
        "inf_net.f_sigma_batchnorm.running_mean": deserializeTensor(modelUpdate.inf_net_f_sigma_batchnorm_running_mean),
        "inf_net.f_sigma_batchnorm.running_var": deserializeTensor(modelUpdate.inf_net_f_sigma_batchnorm_running_var),
        "inf_net.f_sigma_batchnorm.num_batches_tracked": inf_net_f_sigma_batchnorm_num_batches_tracked,
        "beta_batchnorm.running_mean": deserializeTensor(modelUpdate.beta_batchnorm_running_mean),
        "beta_batchnorm.running_var": deserializeTensor(modelUpdate.beta_batchnorm_running_var),
        "beta_batchnorm.num_batches_tracked": beta_batchnorm_num_batches_tracked
    }

    if modelUpdate.HasField("topic_word_matrix"):
        modelStateDict["topic_word_matrix"] = deserializeTensor(
            modelUpdate.topic_word_matrix)

    if modelUpdate.HasField("inf_net_adapt_bert_weight"):
        modelStateDict["inf_net.adapt_bert.weight"] =\
            deserializeTensor(
            modelUpdate.inf_net_adapt_bert_weight)

    if modelUpdate.HasField("inf_net_adapt_bert_bias"):
        modelStateDict["inf_net.adapt_bert.bias"] =\
            deserializeTensor(
            modelUpdate.inf_net_adapt_bert_bias)

    return modelStateDict

def read_config_experiments(file_path, skip=[]):
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
                    config_dict[option] = tuple(map(int, value[1:-1].split(',')))
                else:
                    config_dict[option] = value
            
    return config_dict
