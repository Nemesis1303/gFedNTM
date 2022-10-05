import argparse
import json
import pathlib
import shutil
from pathlib import Path
import sys

import dask.dataframe as dd
import pyarrow as pa
from dask.diagnostics import ProgressBar
from gensim import corpora

path_real = "/export/usuarios_ml4ds/lbartolome/data/training_data"
nw = 0

class textPreproc(object):
    """
    A simple class to carry out some simple text preprocessing tasks
    that are needed by topic modeling
    - Stopword removal
    - Replace equivalent terms
    - Calculate BoW
    - Generate the files that are needed for training of different
      topic modeling technologies

    It allows to use Gensim or Spark functions
    """

    def __init__(self, stw_files=[], eq_files=[],
                 min_lemas=15, no_below=10, no_above=0.6,
                 keep_n=100000, cntVecModel=None,
                 GensimDict=None, logger=None):
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
        cntVecModel : pyspark.ml.feature.CountVectorizerModel
            CountVectorizer Model to be used for the BOW calculation
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
        self._cntVecModel = cntVecModel
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
        trDF: Dask or Spark dataframe
            This routine works on the following column "all_lemmas"
            Other columns are left untouched
        nw: Number of workers to use if Dask is selected
            If nw=0 use Dask default value (number of cores)

        Returns
        -------
        trDFnew: A new dataframe with a new colum bow containing the
        bow representation of the documents
        """
        if isinstance(trDF, dd.DataFrame):

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
                    # lowercase and tokenization (similar to Spark tokenizer)
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
                if nw>0:
                    DFtokens = DFtokens.compute(scheduler='processes', num_workers=nw)
                else:
                    #Use Dask default (i.e., number of available cores)
                    DFtokens = DFtokens.compute(scheduler='processes')
            self._GensimDict = corpora.Dictionary(
                DFtokens['final_tokens'].values.tolist())

            # Remove words that appear in less than no_below documents, or in more than
            # no_above, and keep at most keep_n most frequent terms

            self._logger.info('-- -- Gensim Filter Extremes')

            self._GensimDict.filter_extremes(no_below=self._no_below,
                                             no_above=self._no_above, keep_n=self._keep_n)

            # We skip the calculation of the bow for each document, because Spark LDA will
            # not be used in this case. Note that this is different from what is done for
            # Spark preprocessing
            trDFnew = trDF

        else:
            # Preprocess data using Spark
            # tokenization
            tk = Tokenizer(inputCol="all_lemmas", outputCol="tokens")
            trDF = tk.transform(trDF)

            # Removal of Stopwords - Skip if not stopwords are provided
            # to save computation time
            if len(self._stopwords):
                swr = StopWordsRemover(inputCol="tokens", outputCol="clean_tokens",
                                       stopWords=self._stopwords)
                trDF = swr.transform(trDF)
            else:
                # We need to create a copy of the tokens with the new name
                trDF = trDF.withColumn("clean_tokens", trDF["tokens"])

            # Filter according to number of lemmas in each document
            trDF = trDF.where(F.size(F.col("clean_tokens")) >= self._min_lemas)

            # Equivalences replacement
            if len(self._equivalents):
                df = trDF.select(trDF.id, F.explode(trDF.clean_tokens))
                df = df.na.replace(self._equivalents, 1)
                df = df.groupBy("id").agg(F.collect_list("col"))
                trDF = (trDF.join(df, trDF.id == df.id, "left")
                        .drop(df.id)
                        .withColumnRenamed("collect_list(col)", "final_tokens")
                        )
            else:
                # We need to create a copy of the tokens with the new name
                trDF = trDF.withColumn("final_tokens", trDF["clean_tokens"])

            if not self._cntVecModel:
                cntVec = CountVectorizer(inputCol="final_tokens",
                                         outputCol="bow", minDF=self._no_below,
                                         maxDF=self._no_above, vocabSize=self._keep_n)
                self._cntVecModel = cntVec.fit(trDF)

            trDFnew = (self._cntVecModel.transform(trDF)
                           .drop("tokens", "clean_tokens", "final_tokens")
                       )

        return trDFnew

    def saveCntVecModel(self, dirpath):
        """
        Saves a Count Vectorizer Model to the specified path
        Saves also a text document with the corresponding
        vocabulary

        Parameters
        ----------
        dirpath: pathlib.Path
            The folder where the CountVectorizerModel and the
            text file with the vocabulary will be saved

        Returns
        -------
        status: int
            - 1: If the files were generated sucessfully
            - 0: Error (Count Vectorizer Model does not exist)
        """
        if self._cntVecModel:
            cntVecModel = dirpath.joinpath('CntVecModel')
            if cntVecModel.is_dir():
                shutil.rmtree(cntVecModel)
            self._cntVecModel.save(f"file://{cntVecModel.as_posix()}")
            with dirpath.joinpath('vocabulary.txt').open('w', encoding='utf8') as fout:
                fout.write(
                    '\n'.join([el for el in self._cntVecModel.vocabulary]))
            return 1
        else:
            return 0

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
        trDF: Dask or Spark dataframe
            If Spark, the dataframe should contain a column "bow" that will
            be used to calculate the training data
            If Dask, it should contain a column "final_tokens"
        dirpath: pathlib.Path
            The folder where the data will be saved
        tmTrainer: string
            The output format [mallet|sparkLDA|prodLDA|ctm]
        nw: Number of workers to use if Dask is selected
            If nw=0 use Dask default value (number of cores)

        Returns
        -------
        outFile: Path
            A path containing the location of the training data in the indicated format
        """

        self._logger.info(f'-- -- Exporting corpus to {tmTrainer} format')

        if isinstance(trDF, dd.DataFrame):
            # Dask dataframe

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
                    if nw>0:
                        DFmallet.to_csv(outFile, index=False, header=False, single_file=True,
                                    compute_kwargs={'scheduler': 'processes', 'num_workers': nw})
                    else:
                        #Use Dask default number of workers (i.e., number of cores)
                        DFmallet.to_csv(outFile, index=False, header=False, single_file=True,
                                    compute_kwargs={'scheduler': 'processes'})

            elif tmTrainer == 'sparkLDA':
                self._logger.error(
                    '-- -- sparkLDA requires preprocessing with spark')
                return

            elif tmTrainer == "prodLDA":

                outFile = dirpath.joinpath('corpus.parquet')
                if outFile.is_file():
                    outFile.unlink()

                with ProgressBar():
                    DFparquet = trDF[['id', 'cleantext']].rename(
                        columns={"cleantext": "bow_text"})
                    if nw>0:
                        DFparquet.to_parquet(outFile, write_index=False, compute_kwargs={
                                         'scheduler': 'processes', 'num_workers': nw})
                    else:
                        #Use Dask default number of workers (i.e., number of cores)
                        DFparquet.to_parquet(outFile, write_index=False, compute_kwargs={
                                         'scheduler': 'processes'})

            elif tmTrainer == "ctm":
                outFile = dirpath.joinpath('corpus.parquet')
                if outFile.is_file():
                    outFile.unlink()

                with ProgressBar():
                    # DFparquet = trDF[['id', 'cleantext', 'all_rawtext']].rename(
                    #    columns={"cleantext": "bow_text"})
                    DFparquet = trDF[['id', 'cleantext', 'embeddings']].rename(
                        columns={"cleantext": "bow_text"})
                    schema = pa.schema([
                        ('id', pa.int64()),
                        ('bow_text', pa.string()),
                        ('embeddings', pa.list_(pa.float64()))
                    ])
                    if nw>0:
                        DFparquet.to_parquet(outFile, write_index=False, schema=schema, compute_kwargs={
                                         'scheduler': 'processes', 'num_workers': nw})
                    else:
                        #Use Dask default number of workers (i.e., number of cores)
                        DFparquet.to_parquet(outFile, write_index=False, schema=schema, compute_kwargs={
                                         'scheduler': 'processes'})

        else:
            # Spark dataframe
            if tmTrainer == "mallet":
                # We need to convert the bow back to text, and save text file
                # in mallet format
                outFile = dirpath.joinpath('corpus.txt')
                vocabulary = self._cntVecModel.vocabulary
                spark.sparkContext.broadcast(vocabulary)

                # User defined function to recover the text corresponding to BOW
                def back2text(bow):
                    text = ""
                    for idx, tf in zip(bow.indices, bow.values):
                        text += int(tf) * (vocabulary[idx] + ' ')
                    return text.strip()
                back2textUDF = F.udf(lambda z: back2text(z))

                malletDF = (trDF.withColumn("bow_text", back2textUDF(F.col("bow")))
                            .withColumn("2mallet", F.concat_ws(" 0 ", "id", "bow_text"))
                            .select("2mallet")
                            )
                # Save as text file
                # Ideally everything should get written to one text file directly from Spark
                # but this is failing repeatedly, so I avoid coalescing in Spark and
                # instead concatenate all files after creation
                tempFolder = dirpath.joinpath('tempFolder')
                #malletDF.coalesce(1).write.format("text").option("header", "false").save(f"file://{tempFolder.as_posix()}")
                malletDF.write.format("text").option("header", "false").save(
                    f"file://{tempFolder.as_posix()}")
                # Concatenate all text files
                with outFile.open("w", encoding="utf8") as fout:
                    for inFile in [f for f in tempFolder.iterdir() if f.name.endswith('.txt')]:
                        fout.write(inFile.open("r").read())
                shutil.rmtree(tempFolder)

            elif tmTrainer == "sparkLDA":
                # Save necessary columns for Spark LDA in parquet file
                outFile = dirpath.joinpath('corpus.parquet')
                trDF.select("id", "source", "bow").write.parquet(
                    f"file://{outFile.as_posix()}", mode="overwrite")
            elif tmTrainer == "prodLDA":
                outFile = dirpath.joinpath('corpus.parquet')
                lemas_df = (trDF.withColumn("bow_text", back2textUDF(
                    F.col("bow"))).select("id", "bow_text"))
                lemas_df.write.parquet(
                    f"file://{outFile.as_posix()}", mode="overwrite")
            elif tmTrainer == "ctm":
                outFile = dirpath.joinpath('corpus.parquet')
                lemas_raw_df = (trDF.withColumn("bow_text", back2textUDF(
                    F.col("bow"))).select("id", "bow_text", "embeddings"))
                lemas_raw_df.write.parquet(
                    f"file://{outFile.as_posix()}", mode="overwrite")

##############################################################################
#                                  MAIN                                      #
##############################################################################
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Topic modeling utilities')
    parser.add_argument('--spark', action='store_true', default=False,
                        help='Indicate that spark cluster is available',
                        required=False)
    parser.add_argument('--preproc', action='store_true', default=False,
                        help="Preprocess training data according to config file")
    parser.add_argument('--config', type=str, default=None,
                    help="path to configuration file")
    args = parser.parse_args()

    if args.spark:
        # Spark imports and session generation
        import pyspark.sql.functions as F
        from pyspark.ml.feature import (CountVectorizer, StopWordsRemover,
                                        Tokenizer)
        from pyspark.sql import SparkSession

        spark = SparkSession\
            .builder\
            .appName("Topicmodeling")\
            .getOrCreate()

    else:
        spark = None

    # If the preprocessing flag is activated, we need to check availability of
    # configuration file, and run the preprocessing of the training data using
    # the textPreproc class
    if args.preproc:

        configFile = Path(args.config)
        if configFile.is_file():
            with configFile.open('r', encoding='utf8') as fin:
                train_config = json.load(fin)

            """
            Data preprocessing This part of the code will preprocess all the
            documents that are available in the training dataset and generate
            also the necessary objects for preprocessing objects during inference
            """

            tPreproc = textPreproc(stw_files=train_config['Preproc']['stopwords'],
                                    eq_files=train_config['Preproc']['equivalences'],
                                    min_lemas=train_config['Preproc']['min_lemas'],
                                    no_below=train_config['Preproc']['no_below'],
                                    no_above=train_config['Preproc']['no_above'],
                                    keep_n=train_config['Preproc']['keep_n'])

            # Create a Dataframe with all training data
            trDtFile = Path(train_config['TrDtSet'])
            with trDtFile.open() as fin:
                trDtSet = json.load(fin)
        
        if args.spark:
            # Read all training data and configure them as a spark dataframe
            for idx, DtSet in enumerate(trDtSet['Dtsets']):
                df = spark.read.parquet(f"file://{DtSet['parquet']}")
                if len(DtSet['filter']):
                    # To be implemented
                    # Needs a spark command to carry out the filtering
                    # df = df.filter ...
                    pass
                df = (
                    df.withColumn("all_lemmas", F.concat_ws(
                        ' ', *DtSet['lemmasfld']))
                        .withColumn("source", F.lit(DtSet["source"]))
                        .select("id", "source", "all_lemmas")
                )
                if idx == 0:
                    trDF = df
                else:
                    trDF = trDF.union(df).distinct()
            

            # We preprocess the data and save the CountVectorizer Model used to obtain the BoW
            trDF = tPreproc.preprocBOW(trDF)
            tPreproc.saveCntVecModel(configFile.parent.resolve())

            # print("LLEGA 1")
            # # If the trainer is CTM, we also need the embeddings
            # # We get full df containing the embeddings
            # for idx, DtSet in enumerate(trDtSet['Dtsets']):
            #     df = spark.read.parquet(f"file://{DtSet['parquet']}")
            #     df = df.select("id", "embeddings", "fieldsOfStudy")
            #     if idx == 0:
            #         eDF = df
            #     else:
            #         eDF = eDF.union(df).distinct()
            # print("LLEGA 2")
            # # We perform a left join to keep the embeddings of only those documents kept after preprocessing
            # # TODO: Check that this is done properly in Spark
            # trDF = (trDF.join(eDF, trDF.id == eDF.id, "left")
            #         .drop(df.id))

            trDataFile = tPreproc.exportTrData(trDF=trDF,
                                                dirpath=pathlib.Path(path_real),
                                                tmTrainer='ctm')
            sys.stdout.write(trDataFile.as_posix())
        
        else:
            # Read all training data and configure them as a dask dataframe
            for idx, DtSet in enumerate(trDtSet['Dtsets']):

                df = dd.read_parquet(DtSet['parquet']).fillna("")

                # Concatenate text fields
                for idx2, col in enumerate(DtSet['lemmasfld']):
                    if idx2 == 0:
                        df["all_lemmas"] = df[col]
                    else:
                        df["all_lemmas"] += " " + df[col]
                df["source"] = DtSet["source"]
                df = df[["id", "source", "all_lemmas"]]

                # Concatenate dataframes
                if idx == 0:
                    trDF = df
                else:
                    trDF = dd.concat([trDF, df])

            trDF = tPreproc.preprocBOW(trDF, nw)
            tPreproc.saveGensimDict(pathlib.Path(path_real))

            # We get full df containing the embeddings
            for idx, DtSet in enumerate(trDtSet['Dtsets']):
                df = dd.read_parquet(DtSet['parquet']).fillna("")
                df = df[["id", "embeddings", "fieldsOfStudy"]]

                # Concatenate dataframes
                if idx == 0:
                    eDF = df
                else:
                    eDF = dd.concat([trDF, df])

            # We perform a left join to keep the embeddings of only those documents kept after preprocessing
            trDF = trDF.merge(eDF, how="left", on=["id"])

            trDataFile = tPreproc.exportTrData(trDF=trDF,
                                            dirpath=pathlib.Path(path_real),
                                            tmTrainer="ctm",
                                            nw=nw)
    print("Preprocessed file save: ", trDataFile.as_posix())