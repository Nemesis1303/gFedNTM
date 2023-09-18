"""
Carries out specific preprocessing for TM.
"""
import argparse
import datetime as DT
import os
import pathlib
import warnings

warnings.filterwarnings(action="ignore")
from src.tmWrapper import TMWrapper

def select_file_from_directory(directory_path: str):
    """Prints the files in a directory and prompts the user to select one of them.
    """
    files = os.listdir(directory_path)
    # Print the list of available stopwords/equivalences
    print("Available stopwords/equivalences:")
    for index, file_name in enumerate(files):
        print(f"{index + 1}. {file_name}")

    # Prompt the user for input
    while True:
        selection = input(
            "Enter the numbers of the files you want to select (comma-separated): ")
        selections = selection.split(",")
        selected_files = []
        try:
            for s in selections:
                file_index = int(s.strip())
                if 1 <= file_index <= len(files):
                    selected_files.append(os.path.join(directory_path, files[file_index - 1]))
                else:
                    print(f"Invalid selection: {file_index}. Ignoring.")
            break
        except ValueError:
            print("Invalid input. Please enter comma-separated numbers.")

    return selected_files


def get_preproc_params(wordlists_path: str):
    preproc = {
        "min_lemas": 15,
        "no_below": 15,
        "no_above": 0.4,
        "keep_n": 100000,
        "stopwords": [],
        "equivalences": []
    }

    # Get stopwords from selected files
    print("#"*40)
    print("Select stopwords files:")
    print("#"*40)
    preproc["stopwords"]  = select_file_from_directory(wordlists_path)

    # Get equivalences from selected files
    print("\n")
    print("#"*40)
    print("Select equivalences files:")
    print("#"*40)
    preproc["equivalences"] = select_file_from_directory(wordlists_path)
       
    # Prompt user to modify initial values
    preproc["min_lemas"] = int(input(
        "Enter the minimum lemmas value (default, 15): ") or preproc["min_lemas"])
    preproc["no_below"] = int(input(
        "Enter the no_below value (default, 15): ") or preproc["no_below"])
    preproc["no_above"] = float(input(
        "Enter the no_above value (default, 0.4): ") or preproc["no_above"])
    preproc["keep_n"] = int(input(
        "Enter the keep_n value (default, 100000):") or preproc["keep_n"])

    return preproc


def main(path_preproc: str,
         wordlists_path: str,
         parquetFile: str,
         idfld: str,
         trainer: str,
         nw=0,
         iter_=0):

    # Create preproc corpora folder
    models = pathlib.Path(path_preproc)
    models.mkdir(parents=True, exist_ok=True)
    model_path = models.joinpath("iter_" + str(iter_))

    # Get preproc parameters
    Preproc = get_preproc_params(wordlists_path)

    # Create configuration dictionary for dataset
    Dtset = pathlib.Path(parquetFile).stem
    TrDtset = {
        "name": Dtset,
        "Dtsets": [
            {
                "parquet": parquetFile,
                "source": Dtset,
                "idfld": idfld,
                "lemmasfld": [
                    "lemmas"
                ],
                "filter": ""
            }
        ]
    }

    # Create training (preprocessing) configuration dictionary
    train_config = {
        "name": Dtset,
        "description": "",
        "visibility": "Public",
        "trainer": trainer,
        "Preproc": Preproc,
        "TMparam": {},
        "creation_date": DT.datetime.now(),
        "hierarchy-level": 0,
        "htm-version": None,
    }

    # Preprocess
    tm_wrapper = TMWrapper()
    tm_wrapper.preproc_corpus_tm(
        path_preproc=model_path,
        Dtset=Dtset,
        TrDtset=TrDtset,
        train_config=train_config,
        nw=nw,
    )

    return


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Preprocessing for TM')
    parser.add_argument('--path_preproc', type=str,
                        default="/export/usuarios_ml4ds/lbartolome/Datasets/S2CS-AI/models_preproc",
                        help="Path to the folder to save the preprocessed training datasets in the topicmodeler format.")
    parser.add_argument('--parquetFile', type=str,
                        default="/export/clusterdata/jarenas/Datasets/semanticscholar/20230418/parquet/papers_AI_Kwds_NLP_embeddings.parquet",
                        help="Path to corpus to preprocess.")
    parser.add_argument('--idfld', type=str, required=False, 
                        default="corpusid",
                        help="Field to use as id.")
    parser.add_argument('--trainer', type=str, required=False, 
                        default="mallet",
                        help="TM Trainer to use.")
    parser.add_argument('--nw', type=int, required=False, default=0,
                        help="Number of workers when preprocessing data with Dask. Use 0 to use Dask default")
    parser.add_argument('--iter_', type=int, required=False, default=0,
                        help="Preprocessing number of this file.")
    parser.add_argument('--pathWordlists', type=str, required=False,    
                        default="src/preprocessing/wordlists",
                        help="Path to wordlists folder.")

    args = parser.parse_args()
    
    main(path_preproc=args.path_preproc,
        wordlists_path=args.pathWordlists,
        parquetFile=args.parquetFile,
        idfld=args.idfld,
        trainer=args.trainer,
        nw=args.nw,
        iter_=args.iter_)
