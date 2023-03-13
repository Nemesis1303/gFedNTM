# -*- coding: utf-8 -*-
"""
Created on March 3, 2023

.. codeauthor:: L. Calvo-Bartolomé (lcalvo@pa.uc3m.es)
"""

import argparse
import json
import logging
import pickle
import shutil
from pathlib import Path

import colored
import numpy as np
import pandas as pd
import torch
from scipy.special import softmax
from sklearn.preprocessing import normalize
from tqdm import tqdm

from src.models.base.pytorchavitm.avitm_network.avitm import AVITM
from src.models.base.pytorchavitm.datasets.bow_dataset import BOWDataset
from src.models.base.pytorchavitm.utils.data_preparation import prepare_dataset


def printgr(text: str) -> str:
    """    
    Prints the text string given as input in green.

    :param text: Text to print
    :type text: str
    :return: Text to print in green
    :rtype: str
    """
    print(colored.stylize(text, colored.fg('green')))


def rotateArray(arr: np.ndarray, n: int, d: int) -> np.ndarray:
    """Rotates an array of size n by d elements. 

    :param arr: Array to rotate
    :type arr: np.ndarray
    :param n: Size of the array
    :type n: int
    :param d: Number of positions to rotate the array
    :type d: int
    :return: The rotated array
    :rtype: np.ndarray
    """

    temp = []
    i = 0
    while (i < d):
        temp.append(arr[i])
        i = i + 1
    i = 0
    while (d < n):
        arr[i] = arr[d]
        i = i + 1
        d = d + 1
    arr[:] = arr[: i] + temp

    return arr


def generateSynthetic(just_inf: bool,
                      gen_docs: bool,
                      vocab_size: int,
                      n_topics: int,
                      beta: float,
                      alpha: int,
                      n_docs: int,
                      n_docs_inf: int,
                      n_docs_global_inf: int,
                      nwords: tuple,
                      alg: str,
                      n_nodes: int,
                      frozen_topics: int,
                      prior_frozen: list,
                      own_topics: int,
                      prior_nofrozen: list) -> tuple[np.ndarray, list, list]:
    """It generates a synthetic dataset for each of the nodes (the generation of just the topic vectors can also be carried out standalone). Documents therein have been generated using a vocabulary of vocab_size terms with no semantic meaning (e.g., term317, term56314) from a given LDA or ProdLDA generative model. The corpus consists of n_docs training +  n_docs_inf documents, or just n_docs_global_inf inference documents, each of them with a random length drawn uniformly in the range given by nwords .

    :param just_inf: Boolean specifying whether the documents to be generated are going to be used just for inference, or both training and inference
    :type just_inf: bool
    :param gen_docs: Boolean specifying whether the documents need to be generated or it is enough with just the topic vectors
    :type gen_docs: bool
    :param vocab_size: Size of the vocabulary with which the documents will be generated
    :type vocab_size: int
    :param n_topics: Number of topics in the LDA/ProdLDA model
    :type n_topics: int
    :param beta: Prior Dirichlet parameter for the topics
    :type beta: float
    :param n_docs: Number of training documents to be generated
    :type n_docs: int
    :param n_docs_inf: Number of documents for inference to be generated if both training and inference documents are being generated
    :type n_docs_inf: int
    :param n_docs_global_inf: Number of documents for inference when no training documents are being generated
    :type n_docs_global_inf: int
    :param nwords: Tuple with the information of the maximum and minimum number of words a document must have
    :type nwords: tuple
    :param alg: Algorithm (LDA/ProdLDA) that will be used for the documents' generation
    :type alg: str
    :param n_nodes: Number of nodes in the federation
    :type n_nodes: int
    :param prior_frozen: Documents prior of the frozen topics
    :type prior_frozen: list
    :param own_topics: Number of own topics each node has
    :type own_topics: int
    :param prior_nofrozen: Documents prior of the non-frozen topics
    :type prior_nofrozen: list
    :return: three lists, with the word-topic distribution, and a list of document-topic distriubtions and the documents itself, for each of the nodes in the generation
    :rtype: tuple[np.ndarray,list,list]
    """

    if just_inf:
        n_total_docs = n_docs_global_inf
    else:
        n_total_docs = n_docs + n_docs_inf

    # Step 1 - generation of topics
    topic_vectors = np.random.dirichlet(vocab_size*[beta], n_topics)

    # Step 2 - generation of document topic proportions
    doc_topics_all = []
    for i in np.arange(n_nodes):
        doc_topics = np.random.dirichlet(
            prior_frozen + prior_nofrozen, n_total_docs)
        prior_nofrozen = rotateArray(
            prior_nofrozen, len(prior_nofrozen), own_topics)
        doc_topics_all.append(doc_topics)

    # Step 3 - Document generation
    documents_all = []
    # z_all = []

    if gen_docs:
        for i in np.arange(n_nodes):
            print("Generating document words for node ", str(i))
            documents = []  # Document words
            # z = [] # Assignments
            for docid in tqdm(np.arange(n_total_docs)):
                doc_len = np.random.randint(low=nwords[0], high=nwords[1])
                this_doc_words = []
                #this_doc_assigns = []
                for wd_idx in np.arange(doc_len):

                    tpc = np.nonzero(np.random.multinomial(
                        1, doc_topics_all[i][docid]))[0][0]
                    # this_doc_assigns.append(tpc)
                    if alg == "lda":
                        word = np.nonzero(np.random.multinomial(
                            1, topic_vectors[tpc]))[0][0]
                    else:  # prodlda
                        pval = np.power(
                            topic_vectors[tpc], doc_topics_all[i][docid][tpc])
                        # create a tensor of weights
                        weights = torch.tensor(pval, dtype=torch.float)
                        word = torch.multinomial(weights, 1).numpy()[0]
                    this_doc_words.append('wd'+str(word))
                # z.append(this_doc_assigns)
                documents.append(this_doc_words)
            documents_all.append(documents)
            # z_all.append(z)

    return topic_vectors, doc_topics_all, documents_all


def create_model_folder(modelname, modelsdir):
    """_summary_

    :param modelname: _description_
    :type modelname: _type_
    :param modelsdir: _description_
    :type modelsdir: _type_
    :return: _description_
    :rtype: _type_
    """

    # Create model folder and save model training configuration
    modeldir = modelsdir.joinpath(modelname)

    if modeldir.exists():

        # Remove current backup folder, if it exists
        old_model_dir = Path(str(modeldir) + '_old/')
        if old_model_dir.exists():
            shutil.rmtree(old_model_dir)

        # Copy current model folder to the backup folder.
        shutil.move(modeldir, old_model_dir)
        print(f'-- -- Creating backup of existing model in {old_model_dir}')

    modeldir.mkdir()
    configFile = modeldir.joinpath('trainconfig.json')

    return modeldir, configFile


def convert_topic_word_to_init_size(vocab_size, model, model_type,
                                    ntopics, id2token, all_words, betas):
    """It converts the topic-word distribution matrix obtained from the training of a model into a matrix with the dimensions of the original topic-word distribution, assigning zeros to those words that are not present in the corpus. 
    It is only of use in case we are training a model over a synthetic dataset, so as to later compare the performance of the attained model in what regards to the similarity between the original and the trained model.

    :param vocab_size: Size of the synethic'data vocabulary
    :type vocab_size: int
    :param model: Model whose topic-word matrix is being transformed
    :type model: _type_
    :param model_type: Type of the trained model (e.g. AVITM)
    :type model_type: str
    :param ntopics: Number of topics of the trained model
    :type ntopics: int
    :param id2token: Mappings with the content of the document-term matrix
    :type id2token: List[tuple]
    :param all_words: List of all the words of the vocabulary of size vocab_size
    :type all_words: List[str]
    :return: Normalized transormed topic-word distribution
    :rtype: ndarray
    """

    if model_type == "avitm":
        w_t_distrib = np.zeros((ntopics, vocab_size), dtype=np.float64)
        wd = model.get_topic_word_distribution()
        wd = softmax(betas, axis=1)
        for i in np.arange(ntopics):
            for idx, word in id2token.items():
                for j in np.arange(len(all_words)):
                    if all_words[j] == word:
                        w_t_distrib[i, j] = wd[i][idx]
                        break
        normalized_array = normalize(w_t_distrib, axis=1, norm='l1')
        return normalized_array
    else:
        print("Method not impleemnted for the selected model type")
        return None


def train_avitm(modelname, modelsdir, corpus, n_topics, logger):
    """_summary_

    :param modelname: _description_
    :type modelname: _type_
    :param modelsdir: _description_
    :type modelsdir: _type_
    :param corpus: _description_
    :type corpus: _type_
    :return: _description_
    :rtype: _type_
    """

    # Create model folder
    modeldir, configFile = create_model_folder(modelname, modelsdir)

    # Create corpus in ProdLDA format (BoWDataset)
    train_data, val_data, input_size, id2token, _, cv = \
        prepare_dataset(corpus)
    idx2token = train_data.idx2token

    avitm = AVITM(logger=logger,
                  input_size=input_size,
                  n_components=n_topics,
                  model_type="prodLDA",
                  hidden_sizes=(100, 100),
                  activation='softplus',
                  dropout=0.2,
                  learn_priors=True,
                  batch_size=64,
                  lr=2e-3,
                  momentum=0.99,
                  solver='adam',
                  num_epochs=100,
                  reduce_on_plateau=False,
                  topic_prior_mean=0.0,
                  topic_prior_variance=None,
                  num_samples=20,
                  num_data_loader_workers=0,
                  verbose=True)

    avitm.fit(train_data, val_data)

    return modeldir, avitm, cv, id2token, idx2token


def eval_betas(beta, topic_vectors):
    """_summary_

    :param beta: _description_
    :type beta: _type_
    :param topic_vectors: _description_
    :type topic_vectors: _type_
    :return: _description_
    :rtype: _type_
    """
    print('Tópicos (equivalentes) evaluados correctamente:')
    score = np.sum(np.max(np.sqrt(beta).dot(np.sqrt(topic_vectors.T)), axis=0))
    printgr(score)
    return score


def eval_thetas(thetas_theoretical, thetas_actual, n_docs):
    """_summary_

    :param thetas_theoretical: _description_
    :type thetas_theoretical: _type_
    :param thetas_actual: _description_
    :type thetas_actual: _type_
    :param n_docs: _description_
    :type n_docs: _type_
    :return: _description_
    :rtype: _type_
    """
    sim_mat_theoretical = np.sqrt(thetas_theoretical).dot(
        np.sqrt(thetas_theoretical.T))
    sim_mat_actual = np.sqrt(thetas_actual).dot(np.sqrt(thetas_actual.T))
    print('Difference in evaluation of doc similarity:')
    score = np.sum(np.abs(sim_mat_theoretical - sim_mat_actual))/n_docs
    printgr(score)
    return score


def run_iter_simulation(result_iters,
                        tm_settings,
                        centralized_settings,
                        logger):

    # Baseline doc-topics generation
    topic_vectors, doc_topics_all, _ = generateSynthetic(
        True, False, **tm_settings, **centralized_settings)

    for i in range(len(doc_topics_all)):
        if i == 0:
            thetas_bas = doc_topics_all[i]
        else:
            thetas_bas = np.concatenate(
                (thetas_bas, doc_topics_all[i]))
    print("Shape of thetas_bas", str(thetas_bas.shape))

    # Generate documents
    topic_vectors, doc_topics_all, documents_all = \
        generateSynthetic(
            False, True, **tm_settings, **centralized_settings)

    # Generate inference corpus and its docs_topics
    inf = \
        [doc for docs_node in documents_all
            for doc in docs_node[tm_settings['n_docs']:(tm_settings['n_docs']+tm_settings['n_docs_global_inf'])]]
    print("Length of the inference corpus ", str(len(inf)))

    for i in range(len(doc_topics_all)):
        if i == 0:
            inf_doc_topics = doc_topics_all[i][tm_settings['n_docs']:(
                tm_settings['n_docs']+tm_settings['n_docs_global_inf'])]
        else:
            inf_doc_topics = np.concatenate(
                (inf_doc_topics, doc_topics_all[i][tm_settings['n_docs']:(tm_settings['n_docs']+tm_settings['n_docs_global_inf'])]))
    print("Shape of inf_doc_topics", str(inf_doc_topics.shape))

    # Here we compare alignment of the topic_vector matrix with itself and with another randomly generated matrix
    topic_vectors2 = np.random.dirichlet(
        tm_settings['vocab_size']*[eta], tm_settings['n_topics'])
    betas_bas = eval_betas(topic_vectors2, topic_vectors)
    result_iters['baseline']['betas'].append(betas_bas)

    ########################
    # Centralized training #
    ########################
    print("CENTRALIZED")
    # Define corpus
    corpus = [
        doc for docs_node in documents_all for doc in docs_node[0:tm_settings['n_docs']]]
    print("Size of centralized corpus ", str(len(corpus)))

    # Train model
    modelname = "prod_centralized"
    modeldir, avitm, cv, id2token, idx2token = train_avitm(
        modelname, modelsdir, corpus, tm_settings['n_topics'], logger)

    # Get betas
    betas = avitm.get_topic_word_distribution()
    betas = softmax(betas, axis=1)
    print("MAX BETAS: ", np.max(betas))
    print("MIN BETAS: ", np.min(betas))
    all_words = \
        ['wd' + str(word) for word in
            np.arange(tm_settings['vocab_size']+1) if word > 0]
    betas = \
        convert_topic_word_to_init_size(
            vocab_size=tm_settings['vocab_size'],
            model=avitm,
            model_type="avitm",
            ntopics=tm_settings['n_topics'],
            id2token=id2token,
            all_words=all_words,
            betas=betas)

    # Eval betas
    betas_31 = eval_betas(betas, topic_vectors)
    result_iters['centralized']['betas'].append(betas_31)

    # Inference: get inferred thetas
    docs_val_conv = [" ".join(inf[i]) for i in np.arange(len(inf))]
    val_bow = cv.transform(docs_val_conv)
    val_bow = val_bow.toarray()
    val_data = BOWDataset(val_bow, idx2token)
    thetas_inf = \
        np.asarray(avitm.get_doc_topic_distribution(val_data))
    thetas_theoretical = inf_doc_topics

    # Eval thetas
    thetas_312 = eval_thetas(
        thetas_theoretical, thetas_inf, len(thetas_inf))
    result_iters['centralized']['thetas'].append(thetas_312)

    #############################
    # Non-colaborative training #
    #############################
    betas_nodes = []
    thetas_nodes = []
    for node in range(centralized_settings['n_nodes']):
        print("NON-COLLABORATIVE of node ", str(node))
        # Define corpus
        corpus = documents_all[node][0:tm_settings['n_docs']]
        print("Size of non-collaborative corpus ", str(len(corpus)))

        # Train model
        modelname = "prodlda_node"
        modeldir, avitm, cv, id2token, idx2token = train_avitm(
            modelname, modelsdir, corpus, tm_settings['n_topics'], logger)

        # Get betas
        betas = avitm.get_topic_word_distribution()
        betas = softmax(betas, axis=1)
        all_words = \
            ['wd'+str(word)
                for word in np.arange(tm_settings['vocab_size']+1)
                if word > 0]
        betas = \
            convert_topic_word_to_init_size(
                vocab_size=tm_settings['vocab_size'],
                model=avitm,
                model_type="avitm",
                ntopics=tm_settings['n_topics'],
                id2token=id2token,
                all_words=all_words,
                betas=betas)

        # Eval betas
        betas_32 = eval_betas(betas, topic_vectors)
        betas_nodes.append(betas_32)

        # Inference: get inferred thetas
        docs_val_conv = \
            [" ".join(inf[i]) for i in np.arange(len(inf))]
        val_bow = cv.transform(docs_val_conv)
        val_bow = val_bow.toarray()
        val_data = BOWDataset(val_bow, idx2token)
        thetas_inf = np.asarray(avitm.get_doc_topic_distribution(
            val_data))  
        thetas_theoretical = inf_doc_topics

        # Eval thetas
        thetas_322 = eval_thetas(
            thetas_theoretical, thetas_inf, len(thetas_inf))
        thetas_nodes.append(thetas_322)

    # Calculate average among nodes
    avg1 = sum(betas_nodes)/centralized_settings['n_nodes']
    avg2 = sum(thetas_nodes)/centralized_settings['n_nodes']
    result_iters['non_colab']['betas'].append(avg1)
    result_iters['non_colab']['thetas'].append(avg2)

    ########################
    #       Baseline       #
    ########################
    print("BASELINE")
    thetas_theoretical = inf_doc_topics
    thetas_baseline = eval_thetas(
        thetas_theoretical, thetas_bas, len(thetas_bas))
    result_iters['baseline']['thetas'].append(thetas_baseline)


def flatten_dict(d, parent_key='', sep='_'):
    """
    Flattens a nested dictionary into a flat dictionary
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Scripts for Embeddings Service")
    parser.add_argument("--path_results", type=str, default="/Users/lbartolome/Documents/GitHub/gFedNTM/experiments_centralized/eta_variable",
                        required=False, metavar=("path_results"),
                        help="Path to folder where the results will be saved")
    args = parser.parse_args()

    logging.basicConfig(level='INFO')
    logger = logging.getLogger('Simulations')

    # Read settings from config file
    experiment_config = Path(args.path_results).joinpath("config.json")
    with experiment_config.open('r', encoding='utf8') as fin:
        modelInfo = json.load(fin)

        # Read folder where the actual models will be stored and the type of experiment that is going to be carried out
        modelsdir = Path(modelInfo['modelsdir'])
        experiment = modelInfo['experiment']

        tm_settings = {
            key: value for key, value in modelInfo.items()
            if key in [
                'vocab_size', 'n_topics', 'beta', 'alpha',
                'n_docs', 'n_docs_inf', 'n_docs_global_inf', 'alg'
            ]
        }
        tm_settings['nwords'] = tuple(modelInfo['nwords'].values())

        # Centralized settings
        prior_frozen = [modelInfo['alpha']] * modelInfo['frozen_topics']
        own_topics = \
            int((modelInfo['n_topics'] - modelInfo['frozen_topics'])
                ) // modelInfo['n_nodes']
        prior_nofrozen = \
            [modelInfo['alpha']] * own_topics + \
            [modelInfo['alpha']/10000] * (modelInfo['n_topics'] -
                                          modelInfo['frozen_topics']-own_topics)

        centralized_settings = {
            "n_nodes": modelInfo['n_nodes'],
            "frozen_topics": modelInfo['frozen_topics'],
            "prior_frozen": prior_frozen,
            "own_topics": own_topics,
            "prior_nofrozen": prior_nofrozen
        }

        # Read lists for experiments
        frozen_topics_list = modelInfo['frozen_topics_list'].split()
        frozen_topics_list = [int(el) for el in frozen_topics_list]
        print(frozen_topics_list)
        eta_list = modelInfo['eta_list'].split()
        eta_list = [float(el) for el in eta_list]
        iters = modelInfo['iters']

    # Dictionary to save the simulation results
    simulations = {
        'centralized':
            {
                'betas_mean': [], 'betas_std': [],
                'thetas_mean': [], 'thetas_std': []
            },
        'non_colab':
            {
                'betas_mean': [], 'betas_std': [],
                'thetas_mean': [], 'thetas_std': []
            },
        'baseline':
            {
                'betas_mean': [], 'betas_std': [],
                'thetas_mean': [], 'thetas_std': []
            }
    }
    
    simulations_keys = ['centralized', 'non_colab', 'baseline']
    stats_keys = ['betas', 'thetas']

    if experiment == 0:
        
        print("experiment is 0")

        for frozen_tpcs_ids in tqdm(range(len(frozen_topics_list))):
            frozen_topics = frozen_topics_list[frozen_tpcs_ids]

            print("Executing for frozen topics ", str(frozen_topics))

            # Create a list of tuples to store the results for each of the iters
            result_iters = {
                'centralized': {'betas': [], 'thetas': []},
                'non_colab': {'betas': [], 'thetas': []},
                'baseline': {'betas': [], 'thetas': []}
            }

            for iter_ in range(iters):
                print(f"Executing for iteration {str(iter_)}")

                # Recalculate centralized settings
                prior_frozen = frozen_topics * [tm_settings['alpha']]
                own_topics = \
                    int((tm_settings['n_topics'] -
                        frozen_topics)/centralized_settings['n_nodes'])
                prior_nofrozen = \
                    own_topics * [tm_settings['alpha']] + \
                    (tm_settings['n_topics']-frozen_topics - own_topics) * \
                    [tm_settings['alpha']/10000]

                centralized_settings = {
                    "n_nodes": centralized_settings['n_nodes'],
                    "frozen_topics": frozen_topics,
                    "prior_frozen": prior_frozen,
                    "own_topics": own_topics,
                    "prior_nofrozen": prior_nofrozen
                }

                # Here we fix eta
                eta = tm_settings["beta"]

                # Execute iteration
                run_iter_simulation(result_iters, tm_settings,
                                    centralized_settings, logger)

            for sim_key in simulations_keys:
                for stat_key in stats_keys:
                    mean_key = f"{stat_key}_mean"
                    std_key = f"{stat_key}_std"
                    simulations[sim_key][mean_key].append(
                        np.mean(result_iters[sim_key][stat_key]))
                    simulations[sim_key][std_key].append(
                        np.std(result_iters[sim_key][stat_key]))

        simulations_flattened = flatten_dict(simulations)
        df = pd.DataFrame(simulations_flattened)
        df = df.set_index(pd.Index(frozen_topics_list))
        df.index.name = 'Nr frozen topics'
        print(df)

    elif experiment == 1:
        print("experiment is 1")

        for eta_id in tqdm(range(len(eta_list))):
            eta = eta_list[eta_id]
            print("Executing for eta equals to ", str(eta))

            # Create a list of tuples to store the results for each of the iters
            result_iters = {
                'centralized': {'betas': [], 'thetas': []},
                'non_colab': {'betas': [], 'thetas': []},
                'baseline': {'betas': [], 'thetas': []}
            }

            for iter_ in range(iters):
                print(f"Executing for iteration {str(iter_)}")

                tm_settings["beta"] = eta

                # Recalculate centralized settings
                frozen_topics = frozen_topics_list[1]
                prior_frozen = frozen_topics * [tm_settings['alpha']]
                own_topics = \
                    int((tm_settings['n_topics'] -
                        frozen_topics)/centralized_settings['n_nodes'])
                prior_nofrozen = \
                    own_topics * [tm_settings['alpha']] + \
                    (tm_settings['n_topics']-frozen_topics - own_topics) * \
                    [tm_settings['alpha']/10000]

                centralized_settings = {
                    "n_nodes": centralized_settings['n_nodes'],
                    "frozen_topics": frozen_topics,
                    "prior_frozen": prior_frozen,
                    "own_topics": own_topics,
                    "prior_nofrozen": prior_nofrozen
                }

                run_iter_simulation(result_iters, tm_settings,
                                    centralized_settings, logger)

            for sim_key in simulations_keys:
                for stat_key in stats_keys:
                    mean_key = f"{stat_key}_mean"
                    std_key = f"{stat_key}_std"
                    simulations[sim_key][mean_key].append(
                        np.mean(result_iters[sim_key][stat_key]))
                    simulations[sim_key][std_key].append(
                        np.std(result_iters[sim_key][stat_key]))

        simulations_flattened = flatten_dict(simulations)
        #import pdb; pdb.set_trace()
        print(simulations_flattened)
        print(eta_list)
        df = pd.DataFrame(simulations_flattened)
        df = df.set_index(pd.Index(eta_list))
        df.index.name = 'Eta'
        print(df)
    # Update where to save
    results_file = Path(args.path_results).joinpath("results.pickle")
    with open(results_file, 'wb') as handle:
        pickle.dump(df, handle)
