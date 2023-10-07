import argparse
import datetime as DT
import logging
import os
import pathlib
import sys
import pandas as pd

# Add src to path and make imports
sys.path.append('../..')
from src.aux_modules.tmWrapper.tm_wrapper import TMWrapper
from src.aux_modules.utils.misc import read_config_experiments

################### LOGGER #################
logger = logging.getLogger()
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
logger.addHandler(handler)
############################################

def train(path_corpus: str,
          models_folder: str,
          trainer: str,
          iters: int,
          start: int,
          training_params: dict,
          ntopics_nodes:int,
          ntopics_centralized:list[int],
          fos_name: str):
    
    path_corpus = pathlib.Path(path_corpus)
    
    tm_wrapper = TMWrapper()
    
    for iter_ in range(iters):
        iter_ += start
        logger.info(f"-- -- Iteration {iter_}")
    
        # Train centralized models
        logger.info(f"{'*'*40}")
        logger.info("-- -- Training centralized models...")
        logger.info(f"{'*'*40}")
        
        ntopics_centralized = [int(x) for x in ntopics_centralized.split(",")]
        for ntopic_centr in ntopics_centralized:
            name = f"centralized_{str(ntopic_centr)}_{str(iter_)}_{DT.datetime.now().strftime('%Y%m%d')}"
            logger.info("-- -- Training centralized model: " + name)
            
            training_params['ntopics'] = ntopic_centr
            model_path = tm_wrapper.train_root_model(
                models_folder=models_folder,
                name=name,
                path_corpus=path_corpus,
                trainer=trainer,
                training_params=training_params,
            )
            
            # Calculate RBO and TD
            tm_wrapper.calculate_rbo(model_path)
            tm_wrapper.calculate_td(model_path)
        
        # Train non-collaborative models
        logger.info(f"{'*'*40}")
        logger.info("-- -- Training non-collaborative models...")
        logger.info(f"{'*'*400}")

        # Read path_corpus to get nodes' corpus by fos and save to file
        df = pd.read_parquet(path_corpus)
        fos = df[fos_name].unique()
        for f in fos:
            
            ntopics_nodes = ntopics_centralized
            
            for ntopic_node in ntopics_nodes:
            
                # Save node corpus to file
                df_f = df[df[fos_name] == f]
                path_node_corpus = pathlib.Path(models_folder).joinpath(f"{f}.parquet")
                print(path_node_corpus)
                df_f.to_parquet(path_node_corpus)
                
                # Train non-collaborative model
                name = f"non_collaborative_{f}_{str(ntopic_node)}_{str(iter_)}_{DT.datetime.now().strftime('%Y%m%d')}"
                logger.info("-- -- Training non-collaborative model: " + name)
                
                training_params['ntopics'] = ntopics_nodes
                model_path = tm_wrapper.train_root_model(
                    models_folder=models_folder,
                    name=name,
                    path_corpus=path_node_corpus,
                    trainer=trainer,
                    training_params=training_params,
                )
                
                # Remove training corpus file
                os.remove(path_node_corpus)
                
                # Calculate RBO and TD
                tm_wrapper.calculate_rbo(model_path)
                tm_wrapper.calculate_td(model_path)
            
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_corpus', type=str,
                        default="/export/usuarios_ml4ds/lbartolome/Repos/my_repos/gFedNTM/static/datasets/dataset_federated/iter_1/corpus.parquet",
                        help="Path to the training data.")
    parser.add_argument('--models_folder', type=str,
                        default="/export/usuarios_ml4ds/lbartolome/Repos/my_repos/gFedNTM/experiments/collab_vs_non_collab/results/models",
                        help="Path where the models are going to be saved.")
    parser.add_argument('--trainer', type=str,
                        default="ctm",
                        help="Name of the underlying topic modeling algorithm to be used: mallet|ctm")
    parser.add_argument('--iters', type=int,
                        default=1,
                        help="Number of iteration to create htms from the same corpus")
    parser.add_argument('--start', type=int,
                        default=0,
                        help="Iter number to start the naming of the root models.")
    parser.add_argument('--ntopics_nodes', type=int, default=5,
                        help="Number of topics in the non-collaborative models.")
    parser.add_argument('--ntopics_centralized', type=str, 
                        default="10,20,30,40,50",
                        help="Number of topics for the centralized models.")
    parser.add_argument('--fos_name', type=str, default="fos",
                    help="Name of the field of study column in the corpus.")
    args = parser.parse_args()
    
    # Read training_params
    config_file = os.path.dirname(os.path.dirname(os.getcwd()))
    if config_file.endswith("gFedNTM"):
        config_file = os.path.join(
                config_file,
                'config',
                'dft_params.cf',
            )
    else:
        config_file = os.path.join(
                config_file,
                'gFedNTM',
                'config',
                'dft_params.cf',
            )
    training_params = read_config_experiments(config_file)

    train(path_corpus=args.path_corpus,
          models_folder=args.models_folder,
          trainer=args.trainer,
          iters=args.iters,
          start=args.start,
          training_params=training_params,
          ntopics_nodes=args.ntopics_nodes,
          ntopics_centralized=args.ntopics_centralized,
          fos_name=args.fos_name)


if __name__ == "__main__":
    main()
