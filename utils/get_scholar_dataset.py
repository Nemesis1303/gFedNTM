import pandas as pd
import numpy as np

file = "/Users/lbartolome/Documents/GitHub/fuzzy-spoon/data/training_data/sample_fields_SS.csv"

df = pd.read_csv(file)
df_filtered = df[["title", "paperAbstract", "fieldsOfStudy"]]
df_fos = df[["fieldsOfStudy"]].drop_duplicates()
pr = df_filtered.loc[df.fieldsOfStudy == "[Biology]",'paperAbstract'].tolist()

df_all = []
for fos in df_fos.values:
    corpus_fos = df_filtered.loc[df.fieldsOfStudy == fos[0],'paperAbstract'].tolist()
    df_all.append(corpus_fos)

df_all_lem = []

np.savez\
 ('/Users/lbartolome/Documents/GitHub/fuzzy-spoon/data/training_data/scholar_25000.npz', n_foss = 25, corpora=df_all, corpora_lem=df_all_lem)