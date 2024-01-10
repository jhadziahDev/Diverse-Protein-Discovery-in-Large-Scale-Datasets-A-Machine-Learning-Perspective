import pandas as pd
from Exploring_data import random_sample
from common import load_embeddings_from_h5


path = r'Dissertation\per_protein.h5'
sample = random_sample(path, 5000)
df = pd.DataFrame(sample)
df_transposed = df.transpose()
df_transposed.to_csv('Proteins_sampled.csv', sep=',', index=True)
print('df has been saved....')

embeddings = load_embeddings_from_h5(path)

emb_df = pd.DataFrame(embeddings)
print(emb_df)
emb_transposed = emb_df.transpose()
print(emb_transposed)

emb_transposed.to_csv('Protein_emb.tsv', sep='\t', index=True)