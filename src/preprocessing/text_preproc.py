import json
import shutil
from pathlib import Path

import dask.dataframe as dd
import pyarrow as pa
from dask.diagnostics import ProgressBar
from gensim import corpora


class textPreproc(object):
    """
    A simple class to carry out some simple text preprocessing tasks
    that are needed by topic modeling
    - Stopword removal
    - Replace equivalent terms
    - Calculate BoW
    - Generate the files that are needed for training of different
      topic modeling technologies
    """

    def __init__(self, stw_files=[], eq_files=[],
                 min_lemas=15, no_below=10, no_above=0.6,
                 keep_n=100000, GensimDict=None, logger=None):
        """
        Initilization Method
        Stopwords and the dictionary of equivalences will be loaded
        during initialization

        Parameters
        ----------
        stw_files: list of str
            List of paths to stopwords files
        eq_files: list of str
            List of paths to equivalent terms files
        min_lemas: int
            Minimum number of lemas for document filtering
        no_below: int
            Minimum number of documents to keep a term in the vocabulary
        no_above: float
            Maximum proportion of documents to keep a term in the vocab
        keep_n: int
            Maximum vocabulary size
        GensimDict : gensim.corpora.Dictionary
            Optimized Gensim Dictionary Object
        logger: Logger object
            To log object activity
        """
        self._stopwords = self._loadSTW(stw_files)
        self._equivalents = self._loadEQ(eq_files)
        self._min_lemas = min_lemas
        self._no_below = no_below
        self._no_above = no_above
        self._keep_n = keep_n
        self._GensimDict = GensimDict

        if logger:
            self._logger = logger
        else:
            import logging
            logging.basicConfig(level='INFO')
            self._logger = logging.getLogger('textPreproc')

    def _loadSTW(self, stw_files):
        """
        Loads all stopwords from all files provided in the argument

        Parameters
        ----------
        stw_files: list of str
            List of paths to stopwords files

        Returns
        -------
        stopWords: list of str
            List of stopwords
        """

        stopWords = []
        for stwFile in stw_files:
            with Path(stwFile).open('r', encoding='utf8') as fin:
                stopWords += json.load(fin)['wordlist']

        return list(set(stopWords))

    def _loadEQ(self, eq_files):
        """
        Loads all equivalent terms from all files provided in the argument

        Parameters
        ----------
        eq_files: list of str
            List of paths to equivalent terms files

        Returns
        -------
        equivalents: dictionary
            Dictionary of term_to_replace -> new_term
        """

        equivalent = {}

        for eqFile in eq_files:
            with Path(eqFile).open('r', encoding='utf8') as fin:
                newEq = json.load(fin)['wordlist']
            newEq = [x.split(':') for x in newEq]
            newEq = [x for x in newEq if len(x) == 2]
            newEq = dict(newEq)
            equivalent = {**equivalent, **newEq}

        return equivalent

    def preprocBOW(self, trDF, nw=0):
        """
        Preprocesses the documents in the dataframe to carry
        out the following tasks
            - Filter out short documents (below min_lemas)
            - Cleaning of stopwords
            - Equivalent terms application
            - BoW calculation

        Parameters
        ----------
        trDF: Dask dataframe
            This routine works on the following column "all_lemmas"
            Other columns are left untouched
        nw: Number of workers to use if Dask is selected
            If nw=0 use Dask default value (number of cores)

        Returns
        -------
        trDFnew: A new dataframe with a new colum bow containing the
        bow representation of the documents
        """

        def tkz_clean_str(rawtext):
            """Function to carry out tokenization and cleaning of text

            Parameters
            ----------
            rawtext: str
                string with the text to lemmatize

            Returns
            -------
            cleantxt: str
                Cleaned text
            """
            if rawtext == None or rawtext == '':
                return ''
            else:
                # lowercase and tokenization
                cleantext = rawtext.lower().split()
                # remove stopwords
                cleantext = [
                    el for el in cleantext if el not in self._stopwords]
                # replacement of equivalent words
                cleantext = [self._equivalents[el] if el in self._equivalents else el
                             for el in cleantext]
            return cleantext

        # Compute tokens, clean them, and filter out documents
        # with less than minimum number of lemmas
        trDF['final_tokens'] = trDF['all_lemmas'].apply(
            tkz_clean_str, meta=('all_lemmas', 'object'))
        trDF = trDF.loc[trDF.final_tokens.apply(
            len, meta=('final_tokens', 'int64')) >= self._min_lemas]

        # Gensim dictionary creation. It persists the created Dataframe
        # to accelerate dictionary calculation
        # Filtering of words is carried out according to provided values
        self._logger.info('-- -- Gensim Dictionary Generation')

        with ProgressBar():
            DFtokens = trDF[['final_tokens']]
            if nw > 0:
                DFtokens = DFtokens.compute(
                    scheduler='processes', num_workers=nw)
            else:
                # Use Dask default (i.e., number of available cores)
                DFtokens = DFtokens.compute(scheduler='processes')
        self._GensimDict = corpora.Dictionary(
            DFtokens['final_tokens'].values.tolist())

        # Remove words that appear in less than no_below documents, or in more than no_above, and keep at most keep_n most frequent terms

        self._logger.info('-- -- Gensim Filter Extremes')

        self._GensimDict.filter_extremes(no_below=self._no_below,
                                         no_above=self._no_above, keep_n=self._keep_n)

        return trDF

    def saveGensimDict(self, dirpath):
        """
        Saves a Gensim Dictionary to the specified path
        Saves also a text document with the corresponding
        vocabulary

        Parameters
        ----------
        dirpath: pathlib.Path
            The folder where the Gensim dictionary and the
            text file with the vocabulary will be saved

        Returns
        -------
        status: int
            - 1: If the files were generated sucessfully
            - 0: Error (Gensim dictionary does not exist)
        """
        if self._GensimDict:
            GensimFile = dirpath.joinpath('dictionary.gensim')
            if GensimFile.is_file():
                GensimFile.unlink()
            self._GensimDict.save_as_text(GensimFile)
            with dirpath.joinpath('vocabulary.txt').open('w', encoding='utf8') as fout:
                fout.write(
                    '\n'.join([self._GensimDict[idx] for idx in range(len(self._GensimDict))]))
            return 1
        else:
            return 0

    def exportTrData(self, trDF, dirpath, tmTrainer, nw=0):
        """
        Exports the training data in the provided dataset to the
        format required by the topic modeling trainer

        Parameters
        ----------
        trDF: Dask dataframe
            It should contain a column "final_tokens"
        dirpath: pathlib.Path
            The folder where the data will be saved
        tmTrainer: string
            The output format [mallet|prod|ctm]
        nw: Number of workers to use if Dask is selected
            If nw=0 use Dask default value (number of cores)

        Returns
        -------
        outFile: Path
            A path containing the location of the training data in the indicated format
        """

        self._logger.info(f'-- -- Exporting corpus to {tmTrainer} format')

        # Remove words not in dictionary, and return a string
        vocabulary = set([self._GensimDict[idx]
                          for idx in range(len(self._GensimDict))])

        def tk_2_text(tokens):
            """Function to filter words not in dictionary, and
            return a string of lemmas 

            Parameters
            ----------
            tokens: list
                list of "final_tokens"

            Returns
            -------
            lemmasstr: str
                Clean text including only the lemmas in the dictionary
            """
            #bow = self._GensimDict.doc2bow(tokens)
            # return ''.join([el[1] * (self._GensimDict[el[0]]+ ' ') for el in bow])
            return ' '.join([el for el in tokens if el in vocabulary])

        trDF['cleantext'] = trDF['final_tokens'].apply(
            tk_2_text, meta=('final_tokens', 'str'))

        if tmTrainer == "mallet":

            outFile = dirpath.joinpath('corpus.txt')
            if outFile.is_file():
                outFile.unlink()

            trDF['2mallet'] = trDF['id'].apply(
                str, meta=('id', 'str')) + " 0 " + trDF['cleantext']

            with ProgressBar():
                #trDF = trDF.persist(scheduler='processes')
                DFmallet = trDF[['2mallet']]
                if nw > 0:
                    DFmallet.to_csv(outFile, index=False, header=False, single_file=True,
                                    compute_kwargs={'scheduler': 'processes', 'num_workers': nw})
                else:
                    # Use Dask default number of workers (i.e., number of cores)
                    DFmallet.to_csv(outFile, index=False, header=False, single_file=True,
                                    compute_kwargs={'scheduler': 'processes'})

        elif tmTrainer == "prod":

            outFile = dirpath.joinpath('corpus.parquet')
            if outFile.is_file():
                outFile.unlink()

            with ProgressBar():
                DFparquet = trDF[['id', 'cleantext']].rename(
                    columns={"cleantext": "bow_text"})
                if nw > 0:
                    DFparquet.to_parquet(outFile, write_index=False, compute_kwargs={
                        'scheduler': 'processes', 'num_workers': nw})
                else:
                    # Use Dask default number of workers
                    # (i.e., number of cores)
                    DFparquet.to_parquet(outFile, write_index=False, compute_kwargs={
                        'scheduler': 'processes'})

        elif tmTrainer == "ctm":
            outFile = dirpath.joinpath('corpus.parquet')
            if outFile.is_file():
                outFile.unlink()

            with ProgressBar():
                DFparquet = trDF[['id', 'cleantext', 'embeddings']].rename(
                    columns={"cleantext": "bow_text"})
                schema = pa.schema([
                    ('id', pa.int64()),
                    ('bow_text', pa.string()),
                    ('embeddings', pa.list_(pa.float64()))
                ])
                if nw > 0:
                    DFparquet.to_parquet(outFile, write_index=False, schema=schema, compute_kwargs={
                        'scheduler': 'processes', 'num_workers': nw})
                else:
                    # Use Dask default number of workers
                    # (i.e., number of cores)
                    DFparquet.to_parquet(outFile, write_index=False, schema=schema, compute_kwargs={
                        'scheduler': 'processes'})

        return outFile
