import nltk
import json

nltk.download("punkt")  # Download the Punkt tokenizer data (if not already downloaded)
from nltk.tokenize import sent_tokenize
from transformers import pipeline
import numpy as np
import pandas as pd
import torch
from argparse import ArgumentParser
from tqdm import trange

from transformers import AutoModelForTokenClassification, AutoTokenizer

from analysis.configs import change_path_to_root
if __name__ == "__main__":
    change_path_to_root()

device = "cuda" if torch.cuda.is_available() else "cpu"


def load_model(model_path):
    print(f"Loading model from {model_path}")
    model = AutoModelForTokenClassification.from_pretrained(model_path).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model.to(device)
    classifier = pipeline("ner", model=model, tokenizer=tokenizer, device=device)
    print(f"Model loaded from {model_path} successfully")
    return classifier


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


def annotate(df_books, classifier, id):
    # sentences is a list of all sentences of a given book
    row = df_books.iloc[id]
    text = row["Content"]
    if pd.isna(text):
        return None
    sentences = sent_tokenize(text)

    # Placeholder for the output data
    output_data = []
    for sentence_num, sentence in enumerate(sentences):
        # Tokenize the sentence using your BERT tokenizer
        tags = classifier(sentence)
        token_data = {
            "sentence": sentence,
            "sentence_num": sentence_num,
            "id": id,
            "book_title": row["Book Title"],
            "age_min": row["Age Min"],
            "age_max": row["Age Max"],
            "word_count": float(row["Word Count"]),
            "tags": tags,
        }
        output_data.append(token_data)
    return output_data


def annotate_dataset_old(df_books, classifier):
    output = []
    for i in trange(len(df_books)):
        data = annotate(df_books, classifier, i)
        if data is None:
            continue
        output.extend(data)
    return output


def annotate_dataset(df_books, classifier):
    # Set number of sentences to be classified at once
    SENTENCE_COUNT = 25

    # Initialize the output list
    output = []
    # sentences is a list of all sentences of a given book
    sentences_all = []
     # Loop through all the books
    for id in trange(len(df_books)):
        # Placeholder for the output data
        data = []
        row = df_books.iloc[id]
        text = row["Content"]
        if pd.isna(text):
            continue
        # sentences is a list of all sentences of a given book
        sentences = sent_tokenize(text)
        sentences_all.extend(sentences)
        for sentence_id, sentence in enumerate(sentences):
            # Tokenize the sentence using your BERT tokenizer
            token_data = {
                "sentence": sentence,
                "sentence_num": sentence_id,
                "id": id,
                "age_min": row["Age Min"],
                "age_max": row["Age Max"],
            }
            if "Title" in row:
                token_data["book_title"] = row["Title"]
            data.append(token_data)
        output.extend(data)

    # Pass the sentences in batches to the classifier
    for idx in trange(0, len(sentences_all), SENTENCE_COUNT):
        tags = classifier(sentences_all[idx:idx+SENTENCE_COUNT])
        for sentence_id, sentence in enumerate(sentences_all[idx:idx+SENTENCE_COUNT]):
            # Tokenize the sentence using your BERT tokenizer
            output[idx+sentence_id]["tags"] = tags[sentence_id]

    return output


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--file_path", type=str, default="data/clean/books.csv")
    parser.add_argument("--model", type=str, default="model/distilbert")
    args = parser.parse_args()
    output_path = args.file_path.split("/")[-1].split(".")[0]

    df_books = pd.read_csv(args.file_path)
    classifier = load_model(args.model)
    output = annotate_dataset(df_books, classifier)
    with open(f"data/annotated/{output_path}.json", "w") as f:
        json.dump(output, f, cls=NpEncoder)
    print("Done")
