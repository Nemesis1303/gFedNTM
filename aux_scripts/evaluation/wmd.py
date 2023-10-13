
import argparse
import sys
import numpy as np
import pandas as pd
import gensim.downloader as api
from tqdm import tqdm
import pathlib

sys.path.append('../..')
from src.aux_modules.topicmodeler.src.topicmodeling.manageModels import TMmodel

def calculate_wmds_centralized_vs_node(path_models: str,
                                       path_save: str,
                                       nr_tpcs_lst: list = [
                                           10, 20, 30, 40, 50],
                                       n_words_lst: list = [
                                           10, 50, 100, 200, 300],
                                       tfidf: bool = True):
    """
    Calculate the WMDs for the models in the path_models folder.
    """

    # Load Word2Vec model
    model = api.load('word2vec-google-news-300')
    # Normalize vectors in the Word2Vec class
    model.init_sims(replace=True)

    for nr_tpcs in tqdm(nr_tpcs_lst):
        # Get node models
        node_models = sorted([el for el in path_models.iterdir()
                              if (el.name.startswith('non_collaborative') and int(el.name.split("_")[-3]) == nr_tpcs)], key=lambda el: el.name)
        index_vals = ['Node ' + str(el+1) for el in range(len(node_models))]

        print(f"-- -- Loaded {len(node_models)} node models:")
        [print(f"-- -- -- {el.name}") for el in node_models]
        
        # Get the centralized models
        centr_models = sorted([el for el in path_models.iterdir()
                               if el.name.startswith('centralized')], key=lambda el: el.name)
        centr_vals = [int(el.name.split("_")[1])
                      for el in path_models.iterdir()
                      if el.name.startswith('centralized')]
        centr_vals.sort()
        centr_vals = [f"Centr {el}" for el in centr_vals]

        print(f"-- -- Loaded {len(centr_models)} centralized models:")
        [print(f"-- -- -- {el.name}") for el in centr_models]

        all_models = node_models + centr_models
        column_values = index_vals + centr_vals

        print("*"*100)
        print(f"-- -- Calculating WMDs ...")
        print("*"*100)
        for n_words in tqdm(n_words_lst):
            # Calculate distances
            distances = np.zeros((len(node_models), len(all_models)))
            for idx_ref, ref_model in enumerate(node_models):
                ref_topics = [el[1].split(', ')
                              for el in TMmodel(ref_model.joinpath('TMmodel')).get_tpc_word_descriptions(n_words=n_words, tfidf=tfidf)]
                for idx_comp, comp_model in enumerate(all_models):
                    comp_topics = [el[1].split(', ') for el in TMmodel(comp_model.joinpath(
                        'TMmodel')).get_tpc_word_descriptions(n_words=n_words, tfidf=tfidf)]
                    all_dist = np.zeros((len(ref_topics), len(comp_topics)))
                    for idx1, tpc1 in enumerate(ref_topics):
                        for idx2, tpc2 in enumerate(comp_topics):
                            all_dist[idx1, idx2] = model.wmdistance(
                                tpc1[:n_words], tpc2[:n_words])
                    distances[idx_ref, idx_comp] = np.mean(
                        np.min(all_dist, axis=1))

            # Create dataframe
            df = pd.DataFrame(data=distances,
                              index=index_vals,
                              columns=column_values)

            this_path_save = path_save.joinpath(
                f"wmds_{nr_tpcs}tpcs_{n_words}_words.csv")
            df.to_csv(this_path_save)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Preprocessing for TM')
    parser.add_argument('--path_models', type=str,
                        default="/export/usuarios_ml4ds/lbartolome/Repos/my_repos/gFedNTM/experiments/collab_vs_non_collab/results/models",
                        help="Path to the folder where the models are saved.")
    parser.add_argument('--path_save', type=str,
                        default="/export/usuarios_ml4ds/lbartolome/Repos/my_repos/gFedNTM/experiments/collab_vs_non_collab/results/wmd_dfs",
                        help="Path to the folder where the results are going to besaved.")
    parser.add_argument('--nr_tpcs_lst', type=str,
                        default="10,20,30,40,50",
                        help="Number of topics to be used in the metrics' extraction.")
    parser.add_argument('--n_words_lst', type=str,
                        default="10,25,50,100,200,300",
                        help="Nr of words to be used for the calculation of the WMD.")
    parser.add_argument('--tfidf', type=bool,
                        default=True,
                        help="Whether to use the tfidf or not.")

    args = parser.parse_args()

    calculate_wmds_centralized_vs_node(
        path_models=pathlib.Path(args.path_models),
        path_save=pathlib.Path(args.path_save),
        nr_tpcs_lst=[int(el) for el in args.nr_tpcs_lst.split(',')],
        n_words_lst=[int(el) for el in args.n_words_lst.split(',')],
        tfidf=args.tfidf
    )
