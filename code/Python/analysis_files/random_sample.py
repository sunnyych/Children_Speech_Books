# randomly sample a portion of sentences to annotate for agreement
# import the data for books

import json
import pandas as pd

with open('classified_generics_update.json', 'r') as file:
    books_data = json.load(file)

# randomly select 100 sentences and human annotate

# Convert to DataFrame
df = pd.DataFrame(books_data)
sampled_df = df.sample(n=100, random_state=1)  

sampled_df = sampled_df[['sentence']].rename(columns={'sentence': 'sentences'})

sampled_df.to_csv('sampled_sentences_generics_features.csv', index=False)
