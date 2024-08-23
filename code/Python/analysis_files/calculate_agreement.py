# read in annotated data

import pandas as pd 
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score
import json

# import the annotated data (with a human_annotation column)
df_annotated = pd.read_csv("sampled_sentences_generics_features_annotated - sampled_sentences_generics_features.csv")

# read in the model-labeled data 
with open('classified_generics_update.json', 'r') as file:
    books_data = json.load(file)

df_books = pd.DataFrame(books_data)

# Remove duplicates in df_books if necessary
df_books_unique = df_books.drop_duplicates(subset='sentence')

# Merge using a left join
df_merged = df_annotated.merge(df_books_unique, on='sentence', how='left')

df_merged.rename(columns={'classification_y': 'gpt_annotation', 'classification_x': 'human_annotation'}, inplace=True)


# Calculate the agreement score
agreement_score = accuracy_score(df_merged['human_annotation'], df_merged['gpt_annotation'])
print("Agreement Score:", agreement_score)

# Generate and display the confusion matrix
conf_matrix = confusion_matrix(df_merged['human_annotation'], df_merged['gpt_annotation'])
print("Confusion Matrix:\n", conf_matrix)

y_true = df_merged['human_annotation']
y_pred = df_merged['gpt_annotation']

precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
