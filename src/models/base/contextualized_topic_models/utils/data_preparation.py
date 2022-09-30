import warnings

import numpy as np
import scipy.sparse
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# Local imports
from ..datasets.dataset import CTMDataset


def get_bag_of_words(data, min_length):
    """
    Creates the bag of words
    """
    vect = [np.bincount(x[x != np.array(None)].astype('int'), minlength=min_length)
            for x in data if np.sum(x[x != np.array(None)]) != 0]

    vect = scipy.sparse.csr_matrix(vect)
    return vect


def bert_embeddings_from_file(text_file, sbert_model_to_load, batch_size=32, max_seq_length=None):
    """
    Creates SBERT Embeddings from an input file, assumes one document per line
    """

    model = SentenceTransformer(sbert_model_to_load)

    if max_seq_length is not None:
        model.max_seq_length = max_seq_length

    with open(text_file, encoding="utf-8") as filino:
        texts = list(map(lambda x: x, filino.readlines()))

    check_max_local_length(max_seq_length, texts)

    return np.array(model.encode(texts, show_progress_bar=True, batch_size=batch_size))


def bert_embeddings_from_list(texts, sbert_model_to_load, batch_size=32, max_seq_length=None):
    """
    Creates SBERT Embeddings from a list
    """
    model = SentenceTransformer(sbert_model_to_load)

    if max_seq_length is not None:
        model.max_seq_length = max_seq_length

    check_max_local_length(max_seq_length, texts)

    return np.array(model.encode(texts, show_progress_bar=True, batch_size=batch_size))


def check_max_local_length(max_seq_length, texts):
    max_local_length = np.max([len(t.split()) for t in texts])
    if max_local_length > max_seq_length:
        warnings.simplefilter('always', DeprecationWarning)
        warnings.warn(f"the longest document in your collection has {max_local_length} words, the model instead "
                      f"truncates to {max_seq_length} tokens.")


def prepare_ctm_dataset(corpus, unpreprocessed_corpus=None, custom_embeddings=None, sbert_model_to_load='paraphrase-distilroberta-base-v1', val_size=0.25, max_seq_length=512):
    """It prepares the training data in the format that is asked as input in a CTM model.

    Parameters
    ----------
    corpus: List[str]
        List of (processed) documents to be used for training of the model
    unpreprocessed_corpus: 
        List of (unpreprocessed) documents to be used for training the model.
        Only required if custom_embeddings are not provided
    custom_embeddings: np.ndarray 
        Custom embeddings
    sbert_model_to_load: str (default='paraphrase-distilroberta-base-v1')
        Model (e.g. paraphrase-distilroberta-base-v1) to be used for generating the embeddings
    val_size: float (default=0.25)
        Percentage of the documents to be used for validation
    max_seq_length: int (default=512)

    Returns
    -------
    training_dataset: CTMDataset
        Training dataset in the required format for AVITM
    validation_dataset: CTMDataset
        Validation dataset in the required format for AVITM
    input_size: int
        Size of the input dimensions of the AVITM model to be trained
    id2token: tuple
        Mappings with the content of each training dataset's document-term matrix.
    qt: TopicModelDataPreparation
        Object with all the information for training the CTM model
    embeddings_train: np.ndarray 
        Train embeddings associated with the datasets
    custom_embeddings:
        All (train and validation) embeddings associated with the dataset
    docs_train: list
        Train documents
    """

    if custom_embeddings is None and unpreprocessed_corpus is None:
        raise TypeError(
            "Custom embeddings or an unpreprocessed corpus to generate the embeddings from must be provided")

    # Create embeddings from text if no custom embeddings are provided
    if custom_embeddings is None:
        docs_conv = [" ".join(unpreprocessed_corpus[i])
                     for i in np.arange(len(unpreprocessed_corpus))]
        custom_embeddings = bert_embeddings_from_list(
            docs_conv, sbert_model_to_load, max_seq_length=max_seq_length)

    # Divide text data and embeddings into training and validation sets
    docs_train, docs_val, embeddings_train, embeddings_val = \
        train_test_split(corpus, custom_embeddings,
                         test_size=val_size, random_state=42)

    # Create a CountVectorizer object to convert a collection of text documents into a matrix of token counts
    cv = CountVectorizer(input='content', lowercase=True,
                         stop_words='english', binary=False)

    #######################################
    # Prepare train dataset in CTM format #
    #######################################
    docs_train_conv = [" ".join(docs_train[i])
                       for i in np.arange(len(docs_train))]

    # Learn the vocabulary dictionary, train_bow = document-term matrix.
    train_bow = cv.fit_transform(docs_train_conv)

    # Array mapping from feature integer indices to feature name.
    idx2token = cv.get_feature_names_out()
    input_size = len(idx2token)
    id2token = {k: v for k, v in zip(range(0, len(idx2token)), idx2token)}

    qt = TopicModelDataPreparation(contextualized_model=sbert_model_to_load)
    qt.vectorizer = cv
    qt.id2token = id2token
    qt.vocab = idx2token
    training_dataset = qt.load(embeddings_train, train_bow, id2token)

    ############################################
    # Prepare validation dataset in CTM format #
    ############################################
    docs_val_conv = [" ".join(docs_val[i]) for i in np.arange(len(docs_val))]
    validation_dataset = qt.transform(
        text_for_bow=docs_val_conv, text_for_contextual=docs_val_conv, custom_embeddings=embeddings_val)

    return training_dataset, validation_dataset, input_size, id2token, qt, embeddings_train, custom_embeddings, docs_train


def prepare_hold_out_dataset(hold_out_corpus, qt, unpreprocessed_ho_corpus=None, embeddings_ho=None, sbert_model_to_load='paraphrase-distilroberta-base-v1', max_seq_length=512):
    """It prepares the holdout data in the format that is asked as input in CTM, based on the TopicModelDataPreparation object generated for the training dataset

    Parameters
    ----------
    hold_out_corpus: List[str]
        List of hold-out lemmatized documents
    qt: TopicModelDataPreparation 
        Object that contains the necessary information for the construction of a CTMDataset
    unpreprocessed_ho_corpus: List[str]
        List of hold-out raw documents
    embeddings_ho:
        Embeddings associated with the hold-out documents
    sbert_model_to_load: str (default='paraphrase-distilroberta-base-v1')
        Model (e.g. paraphrase-distilroberta-base-v1) to be used for generating the embeddings
    max_seq_length: int (default=512)

    Returns
    -------
    ho_data: CTMDataset
        Holdout dataset in the required format for CTM
    """

    if embeddings_ho is None and unpreprocessed_ho_corpus is None:
        raise TypeError(
            "Custom embeddings or an unpreprocessed corpus to generate the embeddings from must be provided")

    # Create embeddings from text if no custom embeddings are provided
    if embeddings_ho is None:
        docs_conv = [" ".join(unpreprocessed_ho_corpus[i])
                     for i in np.arange(len(unpreprocessed_ho_corpus))]
        embeddings_ho = bert_embeddings_from_list(
            docs_conv, sbert_model_to_load, max_seq_length=max_seq_length)

    docs_ho_conv = \
        [" ".join(hold_out_corpus[i]) for i in np.arange(len(hold_out_corpus))]
    ho_data = qt.transform(text_for_bow=docs_ho_conv,
                           text_for_contextual=docs_ho_conv, custom_embeddings=embeddings_ho)

    return ho_data


class TopicModelDataPreparation:

    def __init__(self, contextualized_model=None, show_warning=True, max_seq_length=128):
        self.contextualized_model = contextualized_model
        self.vocab = []
        self.id2token = {}
        self.vectorizer = None
        self.label_encoder = None
        self.show_warning = show_warning
        self.max_seq_length = max_seq_length

    def load(self, contextualized_embeddings, bow_embeddings, id2token, labels=None):
        return CTMDataset(
            X_contextual=contextualized_embeddings, X_bow=bow_embeddings, idx2token=id2token, qt=self, labels=labels)

    def fit(self, text_for_contextual, text_for_bow, labels=None, custom_embeddings=None):
        """
        This method fits the vectorizer and gets the embeddings from the contextual model

        Parameters
        ----------
        text_for_contextual:
            list of unpreprocessed documents to generate the contextualized embeddings
        text_for_bow:
            list of preprocessed documents for creating the bag-of-words
        custom_embeddings:
            np.ndarray type object to use custom embeddings (optional).
        labels:
            list of labels associated with each document (optional).
        """

        if custom_embeddings is not None:
            assert len(text_for_contextual) == len(custom_embeddings)

            if text_for_bow is not None:
                assert len(custom_embeddings) == len(text_for_bow)

            if type(custom_embeddings).__module__ != 'numpy':
                raise TypeError(
                    "contextualized_embeddings must be a numpy.ndarray type object")

        if text_for_bow is not None:
            assert len(text_for_contextual) == len(text_for_bow)

        if self.contextualized_model is None and custom_embeddings is None:
            raise Exception(
                "A contextualized model or contextualized embeddings must be defined")

        # TODO: this count vectorizer removes tokens that have len = 1, might be unexpected for the users
        self.vectorizer = CountVectorizer()

        train_bow_embeddings = self.vectorizer.fit_transform(text_for_bow)

        # if the user is passing custom embeddings we don't need to create the embeddings using the model

        if custom_embeddings is None:
            train_contextualized_embeddings = bert_embeddings_from_list(
                text_for_contextual, sbert_model_to_load=self.contextualized_model, max_seq_length=self.max_seq_length)
        else:
            train_contextualized_embeddings = custom_embeddings
        self.vocab = self.vectorizer.get_feature_names()
        self.id2token = {k: v for k, v in zip(
            range(0, len(self.vocab)), self.vocab)}

        if labels:
            self.label_encoder = OneHotEncoder()
            encoded_labels = self.label_encoder.fit_transform(
                np.array([labels]).reshape(-1, 1))
        else:
            encoded_labels = None
        return CTMDataset(
            X_contextual=train_contextualized_embeddings, X_bow=train_bow_embeddings,
            idx2token=self.id2token, labels=encoded_labels)

    def transform(self, text_for_contextual, text_for_bow=None, custom_embeddings=None, labels=None):
        """
        This method create the input for the prediction. Essentially, it creates the embeddings with the contextualized
        model of choice and with trained vectorizer.

        If text_for_bow is missing, it should be because we are using ZeroShotTM

        Parameters
        ----------
        text_for_contextual:
            list of unpreprocessed documents to generate the contextualized embeddings
        text_for_bow:
            list of preprocessed documents for creating the bag-of-words
        custom_embeddings:
            np.ndarray type object to use custom embeddings (optional).
        labels:
            list of labels associated with each document (optional).
        """

        if custom_embeddings is not None:
            assert len(text_for_contextual) == len(custom_embeddings)

            if text_for_bow is not None:
                assert len(custom_embeddings) == len(text_for_bow)

        if text_for_bow is not None:
            assert len(text_for_contextual) == len(text_for_bow)

        if self.contextualized_model is None:
            raise Exception(
                "You should define a contextualized model if you want to create the embeddings")

        if text_for_bow is not None:
            test_bow_embeddings = self.vectorizer.transform(text_for_bow)
        else:
            # dummy matrix
            if self.show_warning:
                warnings.simplefilter('always', DeprecationWarning)
                warnings.warn(
                    "The method did not have in input the text_for_bow parameter. This IS EXPECTED if you "
                    "are using ZeroShotTM in a cross-lingual setting")

            # we just need an object that is matrix-like so that pytorch does not complain
            test_bow_embeddings = scipy.sparse.csr_matrix(
                np.zeros((len(text_for_contextual), 1)))

        if custom_embeddings is None:
            test_contextualized_embeddings = bert_embeddings_from_list(
                text_for_contextual, sbert_model_to_load=self.contextualized_model, max_seq_length=self.max_seq_length)
        else:
            test_contextualized_embeddings = custom_embeddings

        if labels:
            encoded_labels = self.label_encoder.transform(
                np.array([labels]).reshape(-1, 1))
        else:
            encoded_labels = None

        return CTMDataset(X_contextual=test_contextualized_embeddings, X_bow=test_bow_embeddings,
                          idx2token=self.id2token, qt=self, labels=encoded_labels)
