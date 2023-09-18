import configparser
import json
import os
import colored
import pickle
import pathlib
import pandas as pd


def printgr(text):
    print(colored.stylize(text, colored.fg('green')))


def printred(text):
    print(colored.stylize(text, colored.fg('red')))


def printmag(text):
    print(colored.stylize(text, colored.fg('magenta')))


def is_integer(user_str):
    """Check if input string can be converted to integer
    param user_str: string to be converted to an integer value

    Returns
    -------
    :
        The integer conversion of user_str if possible; None otherwise
    """
    try:
        return int(user_str)
    except ValueError:
        return None


def is_float(user_str):
    """Check if input string can be converted to float
    param user_str: string to be converted to an integer value

    Returns
    -------
    :
        The integer conversion of user_str if possible; None otherwise
    """
    try:
        return float(user_str)
    except ValueError:
        return None


def var_num_keyboard(vartype, default, question):
    """Read a numeric variable from the keyboard

    Parameters
    ----------        
    vartype:
        Type of numeric variable to expect 'int' or 'float'
    default:
        Default value for the variable
    question:
        Text for querying the user the variable value

    Returns
    -------
    :
        The value provided by the user, or the default value
    """
    aux = input(question + ' [' + str(default) + ']: ')
    if vartype == 'int':
        aux2 = is_integer(aux)
    else:
        aux2 = is_float(aux)
    if aux2 is None:
        if aux != '':
            print('The value you provided is not valid. Using default value.')
        return default
    else:
        if aux2 >= 0:
            return aux2
        else:
            print('The value you provided is not valid. Using default value.')
            return default


def var_string_keyboard(option, default, question):
    """Read a string variable from the keyboard

    Parameters
    ----------      
    option: str
        Expected format of the string provided  
    default: str
        Default value for the variable
    question: str
        Text for querying the user the variable value

    Returns
    -------
    :
        The value provided by the user, or the default value
    """

    aux = input(question + ' [' + str(default) + ']: ')
    if option == "comma_separated":
        aux2 = [el.strip() for el in aux.split(',') if len(el)]
        if aux2 is not None and len(aux2) <= 1:
            print('The value you provided is not valid. Using default value.')
            return default
        else:
            aux2 = tuple(map(int, aux[1:-1].split(',')))
    elif option == "bool":
        if aux is not None and aux != "True" and aux != "False":
            print('The value you provided is not valid. Using default value.')
            aux2 = None
            return default
        else:
            aux2 = True if aux == "True" else False
    elif option == "dict":
        if aux != '':
            if aux == "None":
                return default
            else:
                try:
                    aux2 = json.loads(aux)
                except:
                    print('The value you provided is not valid. Using default value.')
                    return default
    elif option == "str":
        aux2 = str(aux) if aux != '' else None
    if aux2 is None:
        if aux != '':
            print('The value you provided is not valid. Using default value.')
        return default
    else:
        return aux2


def request_confirmation(msg="     Are you sure?"):

    # Iterate until an admissible response is got
    r = ''
    while r not in ['yes', 'no']:
        r = input(msg + ' (yes | no): ')

    return r == 'yes'


def query_options(options, msg):
    """
    Prints a heading and the options, and returns the one selected by the user

    Parameters
    ----------
    options:
        Complete list of options
    msg:
        Heading message to be printed before the list of
        available options
    """

    print(msg)

    count = 0
    for n in range(len(options)):
        # Print active options without numbering lags
        print(' {}. '.format(count) + options[n])
        count += 1

    range_opt = range(len(options))

    opcion = None
    while opcion not in range_opt:
        opcion = input('What would you like to do? [{0}-{1}]: '.format(
            str(range_opt[0]), range_opt[-1]))
        try:
            opcion = int(opcion)
        except:
            print('Write a number')
            opcion = None

    return opcion


def format_title(tgt_str):
    # sentences = sent_tokenize(tgt_str)
    # capitalized_title = ' '.join([sent.capitalize() for sent in sentences])
    capitalized_title = tgt_str
    # Quitamos " y retornos de carro
    return capitalized_title.replace('"', '').replace('\n', '')


def unpickler(file: str):
    """Unpickle file"""
    with open(file, 'rb') as f:
        return pickle.load(f)


def pickler(file: str, ob):
    """Pickle object to file"""
    with open(file, 'wb') as f:
        pickle.dump(ob, f)
    return


def mallet_corpus_to_df(corpusFile: pathlib.Path):
    """Converts a Mallet corpus file (i.e., file required for the Mallet import command) to a pandas DataFrame

    Parameters
    ----------
    corpusFile: pathlib.Path
        Path to the Mallet corpus file

    Returns
    -------
    :   pandas.DataFrame
        DataFrame with the corpus
    """

    corpus = [line.rsplit(' 0 ')[1].strip() for line in open(
        corpusFile, encoding="utf-8").readlines()]
    indexes = [line.rsplit(' 0 ')[0].strip() for line in open(
        corpusFile, encoding="utf-8").readlines()]
    corpus_dict = {
        'id': indexes,
        'text': corpus
    }
    return pd.DataFrame(corpus_dict)


def corpus_df_to_mallet(corpus_df: pd.DataFrame,
                        outFile: str,
                        id_column: str = 'id',
                        text_column: str = 'text') -> None:
    """Converts a pandas DataFrame to a Mallet corpus file (i.e., file required for the Mallet import command)

    Parameters
    ----------
    corpus_df: pandas.DataFrame
        DataFrame with the corpus
    outFile: str
        Path to the output file
    id_column: str
        Name of the column with the document ids
    """
    corpus_df['2mallet'] = corpus_df[id_column].astype(
        str) + " 0 " + corpus_df[text_column]
    corpus_df = corpus_df[['2mallet']]
    corpus_df.to_csv(outFile, index=False, header=False)

    return

def read_config_experiments(file_path):
    config = configparser.ConfigParser()
    config.read(file_path)

    # Initialize an empty dictionary
    config_dict = {}

    # Loop through each section in the configuration file
    for section in config.sections():
        # Retrieve the options and values within each section
        options = config.options(section)
    
        # Loop through each option in the section
        for option in options:
            # Retrieve the value of each option
            value = config.get(section, option)
            # Store the option-value pair in the section dictionary
            if option in ['ntopics', 'num_iterations', 'batch_size', 'num_threads', 'optimize_interval', 'num_epochs', 'num_samples', 'num_data_loader_workers']:
                
                config_dict[option] = int(value)
            elif option in ['thetas_thr', 'doc_topic_thr',
                            'alpha', 'dropout_in', 'dropout_out', 'lr',
                            'momentum', 'topic_prior_mean']:
                config_dict[option] = float(value)
            elif option == "labels":
                config_dict[option] = ""
            elif option == "topic_prior_variance":
                config_dict[option] = None
            elif option in ["learn_priors", "reduce_on_plateau"]:
                config_dict[option] = True if value == "True" else False
            elif option == "hidden_sizes":
                config_dict[option] = tuple(map(int, value[1:-1].split(',')))
            else:
                config_dict[option] = value
            
    return config_dict
