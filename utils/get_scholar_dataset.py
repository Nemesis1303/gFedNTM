import pandas as pd
import numpy as np

#file = "/Users/lbartolome/Documents/GitHub/fuzzy-spoon/data/training_data/sample_fields_SS.csv"
file = "/Users/lbartolome/Documents/Doctorate/corpus/sample_lemmas_SS.csv"
df = pd.read_csv(file)
df_fos = df[["fieldsOfStudy"]].drop_duplicates()
pr = df.loc[df.fieldsOfStudy == "['Biology']",'lemmas'].tolist()

df_all = []
for fos in df_fos.values:
    corpus_fos = df.loc[df.fieldsOfStudy == fos[0],'lemmas'].tolist()
    df_all.append(corpus_fos)

print(len(df_all))
np.savez\
 ('/Users/lbartolome/Documents/GitHub/fuzzy-spoon/data/training_data/scholar_25000.npz', n_foss = 25, corpora=df_all, corpora_lem=df_all)