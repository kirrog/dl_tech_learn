import pandas as pd

FILE1 = './frames_embeddings.csv'
FILE2 = './frames_embeddings4.csv'

df1 = pd.read_csv(FILE1, sep=';')
df2 = pd.read_csv(FILE2, sep=';')

df_union = pd.concat([df1, df2]).reset_index(drop=True)
df_union.to_csv("frames_embeddings_4labels.csv", sep=';', index=False)