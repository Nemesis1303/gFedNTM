import json
import os
import pathlib
import shutil
import sys
import time
from subprocess import check_output
import numpy as np

import pandas as pd

from src.topicmodeler.src.topicmodeling.manageModels import TMmodel
from src.utils.misc import mallet_corpus_to_df


class TMWrapper(object):

    def __init__(self, logger=None) -> None:
        if logger:
            self._logger = logger
        else:
            import logging
            logging.basicConfig(level=logging.INFO)
            self._logger = logging.getLogger("TMWrapper")

        if os.path.dirname(os.path.dirname(os.getcwd())).endswith('UserInLoopHTM'):
            self._path_topic_modeler = os.path.join(
                os.path.dirname(os.path.dirname(os.getcwd())),
                'src',
                'topicmodeler',
                'src',
                'topicmodeling',
                'topicmodeling.py'
            )
        else:
            self._path_topic_modeler = os.path.join(
                os.path.dirname(os.path.dirname(os.getcwd())),
                'UserInLoopHTM',
                'src',
                'topicmodeler',
                'src',
                'topicmodeling',
                'topicmodeling.py'
            )
        self._logger.info(self._path_topic_modeler)

    def _get_model_config(self,
                          trainer: str,
                          TMparam: dict,
                          hierarchy_level: int,
                          htm_version: str,
                          expansion_tpc: int,
                          thr: float) -> dict:
        """Gets model configuration based on trainer

        Parameters
        ----------
        trainer : str
            Trainer to use. Either 'mallet' or 'ctm'
        TMparam : dict
            Dictionary with parameters for the trainer
        hierarchy_level : int
            Hierarchy level to use
        htm_version : str
            Version of the hierarchy to use
        expansion_tpc : int
            Number of topics to expand
        thr : float
            Threshold to use for the expansion (if HTM-DS is used)

        Returns
        -------
        params : dict
            Dictionary with the model configuration
        """

        if trainer == 'mallet':

            fields = ["ntopics",
                      "labels",
                      "thetas_thr",
                      "mallet_path",
                      "alpha",
                      "optimize_interval",
                      "num_threads",
                      "num_iterations",
                      "doc_topic_thr",
                      "token_regexp"]
        elif trainer == 'ctm':

            fields = ["ntopics",
                      "thetas_thr",
                      "labels",
                      "model_type",
                      "ctm_model_type",
                      "hidden_sizes",
                      "activation",
                      "dropout_in",
                      "dropout_out",
                      "learn_priors",
                      "lr",
                      "momentum",
                      "solver",
                      "num_epochs",
                      "reduce_on_plateau",
                      "batch_size",
                      "topic_prior_mean",
                      "topic_prior_variance",
                      "num_samples",
                      "num_data_loader_workers"]

        params = {"trainer": trainer,
                  "TMparam": {t: TMparam[t] for t in fields},
                  "hierarchy-level": hierarchy_level,
                  "htm-version": htm_version,
                  "expansion_tpc": expansion_tpc,
                  "thr": thr}

        return params

    def _train_model(self, configFile):
        """Trains a topic model using the topicmodeler package

        Parameters
        ----------
        configFile : pathlib.Path
            Path to the configuration file

        Returns
        -------
        output : str
            Output of the command
        """

        t_start = time.perf_counter()

        cmd = f'python {self._path_topic_modeler} --train --config {configFile.as_posix()}'
        self._logger.info(cmd)
        try:
            self._logger.info(f'-- -- Running command {cmd}')
            output = check_output(args=cmd, shell=True)
        except:
            self._logger.error('-- -- Command execution failed')
        t_end = time.perf_counter()

        t_total = t_end - t_start
        self._logger.info(
            f"Total training time --> {t_total}")

        return

    def _do_preproc(self, configFile, nw):

        cmd = f'python {self._path_topic_modeler} --preproc --config {configFile.resolve().as_posix()} --nw {str(nw)}'
        self._logger.info(cmd)

        t_start = time.perf_counter()
        try:
            self._logger.info(f'-- -- Running command {cmd}')
            output = check_output(args=cmd, shell=True)
        except:
            self._logger.error('-- -- Command execution failed')

        self._logger.info(
            f"Total preprocessing time --> {time.perf_counter() - t_start}")

        return

    def preproc_corpus_tm(self,
                          path_preproc,
                          Dtset,
                          TrDtset,
                          train_config,
                          nw):

        # Create folder structure
        path_preproc.mkdir(parents=True, exist_ok=True)
        model_stats = path_preproc.joinpath("stats")
        model_stats.mkdir(parents=True, exist_ok=True)

        # Save dataset json file
        DtsetConfig = path_preproc.joinpath(Dtset)
        with DtsetConfig.open('w', encoding='utf-8') as outfile:
            json.dump(TrDtset, outfile,
                      ensure_ascii=False, indent=2, default=str)

        # Save training config file
        configFile = path_preproc.joinpath("trainconfig.json")
        train_config["TrDtSet"] = DtsetConfig.resolve().as_posix()
        with configFile.open('w', encoding='utf-8') as outfile:
            json.dump(train_config, outfile,
                      ensure_ascii=False, indent=2, default=str)

        self._do_preproc(configFile, nw)

        return

    def train_root_model(self,
                         models_folder: str,
                         name: str,
                         path_corpus: str,
                         trainer: str,
                         training_params: str) -> pathlib.Path:
        """Trains a (root) topic model according to the algorithm specified by 'trainer'. For doing so, it creates a folder for the model with location 'models_folder/name' and copies the corpus to it. It also creates a config file with the training parameters and an object of the class TMmodel to store the model.

        Parameters
        ----------
        models_folder : str
            Path to the folder where the model will be saved
        name : str
            Name of the model folder which will be save within 'models_folder' to store the model
        path_corpus : str
            Path to the corpus file to be used for training. This corpus has already been preprocessed in the format required by the topicmodeler
        trainer : str
            Trainer to use. Either 'mallet' or 'ctm'
        training_params : dict
            Dictionary with the parameters for the trainer

        Returns
        -------
        model_path : pathlib.Path
            Path to the folder where the model is saved
        """

        # Create folder to save model
        model_path = pathlib.Path(models_folder).joinpath(name)

        if model_path.exists():
            # Remove current backup folder, if it exists
            old_model_dir = pathlib.Path(str(model_path) + '_old/')
            if old_model_dir.exists():
                shutil.rmtree(old_model_dir)

            # Copy current model folder to the backup folder.
            shutil.move(model_path, old_model_dir)
            self._logger.info(
                f'-- -- Creating backup of existing model in {old_model_dir}')

        model_path.mkdir(parents=True, exist_ok=True)

        # Copy training corpus (already preprocessed) to folder
        corpusFile = pathlib.Path(path_corpus)
        if not corpusFile.is_dir() and not corpusFile.is_file:
            sys.exit(
                "The provided corpus file does not exist.")

        if trainer == "ctm":
            self._logger.info(f'-- -- Copying corpus.parquet.')
            if corpusFile.is_dir():
                dest = shutil.copytree(
                    corpusFile, model_path.joinpath("corpus.parquet"))
            else: 
                dest = shutil.copy(corpusFile, model_path.joinpath("corpus.parquet"))
        else:
            dest = shutil.copy(corpusFile, model_path.joinpath("corpus.txt"))
        self._logger.info(f'-- -- Corpus file copied in {dest}')

        # Create train config
        train_config = self._get_model_config(
            trainer=trainer,
            TMparam=training_params,
            hierarchy_level=0,
            htm_version=None,
            expansion_tpc=None,
            thr=None)

        configFile = model_path.joinpath("config.json")
        with configFile.open("w", encoding="utf-8") as fout:
            json.dump(train_config, fout, ensure_ascii=False,
                      indent=2, default=str)

        # Train model
        self._train_model(configFile)

        return model_path

    def train_htm_submodel(self,
                           version: str,
                           father_model_path: pathlib.Path,
                           name: str,
                           trainer: str,
                           training_params: dict,
                           expansion_topic: int,
                           thr: float = None) -> pathlib.Path:
        """Trains a second-level model according to the htm version provided (HTM-WS/DS).

        Parameters
        ----------
        version : str
            HTM version to use. Either 'HTM-WS' or 'HTM-DS'
        father_model_path : pathlib.Path
            Path to the folder where the father model is saved
        name : str
            Name of the model folder which will be save within 'father_model_path' to store the submodel
        trainer : str
            Trainer to use. Either 'mallet' or 'ctm'
        training_params : dict
            Dictionary with the parameters for the trainer
        expansion_topic : int
            Number of topics to expand from the father model
        thr : float, optional
            Threshold to use for the expansion of topics, by default None
            
        Returns
        -------
        model_path : pathlib.Path
            Path to the folder where the submodel is saved
        """

        # Create folder for saving node's outputs
        model_path = pathlib.Path(father_model_path).joinpath(name)

        if model_path.exists():
            # Remove current backup folder, if it exists
            old_model_dir = pathlib.Path(
                str(model_path) + '_old/')
            if old_model_dir.exists():
                shutil.rmtree(old_model_dir)

            # Copy current model folder to the backup folder.
            shutil.move(model_path, old_model_dir)
            print(
                f'-- -- Creating backup of existing model in {old_model_dir}')

        model_path.mkdir(parents=True, exist_ok=True)

        # Get training configuration
        train_config = self._get_model_config(
            trainer=trainer,
            TMparam=training_params,
            hierarchy_level=1,
            htm_version=version,
            expansion_tpc=expansion_topic,
            thr=thr)

        # Create training config file for the submodel
        configFile = model_path.joinpath("config.json")
        with configFile.open("w", encoding="utf-8") as fout:
            json.dump(train_config, fout, ensure_ascii=False,
                      indent=2, default=str)

        # Create submodel training corpus
        cmd = f'python {self._path_topic_modeler} --hierarchical --config {father_model_path.joinpath("config.json").as_posix()} --config_child {configFile.as_posix()}'
        print(cmd)
        try:
            self._logger.info(f'-- -- Running command {cmd}')
            output = check_output(args=cmd, shell=True)
        except:
            self._logger.error('-- -- Command execution failed')

        # Train submodel
        self._train_model(configFile)

        return model_path
    
    def calculate_cohr_vs_ref(self,
                              model_path: pathlib.Path,
                              corpus_val: pathlib.Path) -> None:
        """Calculates the topic coherence of the model with respect to a reference corpus. The model should be given as TMmodel, being model_path the path to the folder where the TMmodel is saved.

        Parameters
        ----------
        model_path : pathlib.Path
            Path to the folder where the TMmodel is saved
        corpus_val : pathlib.Path
            Path to the folder where the validation corpus is saved
        """
        
        corpus_df = mallet_corpus_to_df(corpus_val)
        corpus_df['text'] = corpus_df['text'].apply(lambda x: x.split())
        
        tm = TMmodel(model_path.joinpath("TMmodel"))
        cohr = tm.calculate_topic_coherence(
                metrics=["c_npmi"],
                reference_text=corpus_df.text.values.tolist(),
                aggregated=False,
            )
        
        np.save(model_path.joinpath("TMmodel").joinpath(
            'new_topic_coherence.npy'), cohr)
                
        return
