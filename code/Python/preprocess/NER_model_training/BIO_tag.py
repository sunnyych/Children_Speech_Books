import pandas as pd
import json
from transformers import AutoTokenizer
from argparse import ArgumentParser

from analysis.configs import change_path_to_root

if __name__ == "__main__":
    change_path_to_root()

id2label = {
    0: "O",
    1: "B-Artifacts",
    2: "I-Artifacts",
    3: "B-Behavioral",
    4: "I-Behavioral",
    5: "B-Biological Kind",
    6: "I-Biological Kind",
    7: "B-Mental State",
    8: "I-Mental State",
    9: "B-Non-Living Kind",
    10: "I-Non-Living Kind",
    11: "B-Normative Feature",
    12: "I-Normative Feature",
    13: "B-Perceptual",
    14: "I-Perceptual",
    15: "B-Social Kind/Role",
    16: "I-Social Kind/Role",
}
label2id = {v: k for k, v in id2label.items()}


def get_dataset(file_path):
    raw_dataset = json.load(open(file_path, "r", encoding="utf-8"))
    dataset = {}
    for data in raw_dataset:
        sentence_id, sentence, label = (
            data["sentence_id"],
            data["sentence"],
            data["label"],
        )
        sentence_id = str(sentence_id)
        if label in ["Relation", "Purpose"]:
            continue
        if sentence_id not in dataset:
            dataset[sentence_id] = {
                "sentence": sentence,
                "sentence_id": sentence_id,
                "labels": [],
            }
        dataset[sentence_id]["labels"].append(
            {
                "word": data["word"],
                "label": data["label"],
                "start": data["start"],
                "end": data["end"],
            }
        )
    return dataset


def tokenize(data, tokenizer):
    encoded = tokenizer.encode_plus(data["sentence"], return_offsets_mapping=True)
    ids = encoded["input_ids"]
    offsets = encoded["offset_mapping"]
    tokens = tokenizer.convert_ids_to_tokens(ids)
    result = []
    for i, token in enumerate(tokens):
        result.append(
            {
                "token": token,
                "start_index": offsets[i][0],
                "end_index": offsets[i][1],
                "id": ids[i],
            }
        )
    return result


def most_frequent(list_of_labels):
    return max(set(list_of_labels), key=list_of_labels.count)


def get_clean_label(label):
    if label == "O":
        return label
    else:
        return label.split(" @@@ ")[0]


def generate_labeled_tokens(data, tokenizer):
    text = data["sentence"]
    labels = data["labels"]
    tokens = tokenize(data, tokenizer)

    char_label = ["O"] * len(text)

    for i, span in enumerate(labels):
        label = span["label"]
        start = span["start"]
        end = span["end"]

        char_label[start:end] = [f"{label} @@@ #{i}"] * (end - start)

    for i, token in enumerate(tokens):
        if token["start_index"] != token["end_index"]:
            token["raw_label"] = most_frequent(
                char_label[token["start_index"] : token["end_index"]]
            )
        else:
            token["raw_label"] = "O"
        token["clean_label"] = get_clean_label(token["raw_label"])

    # BIO labels
    for i, token in enumerate(tokens):
        if token["raw_label"] != "O":
            if i == 0:
                token["bio_label"] = "B-" + token["clean_label"]

            else:
                if tokens[i - 1]["raw_label"] == tokens[i]["raw_label"]:
                    token["bio_label"] = "I-" + token["clean_label"]
                else:
                    token["bio_label"] = "B-" + token["clean_label"]
        else:
            token["bio_label"] = "O"
    return tokens


def tokens_to_list(sentence_id, tokens):
    ner_tags = []
    tokens_list = []
    token_ids = []
    label_ids = []
    for token in tokens:
        tokens_list.append(token["token"])
        token_ids.append(token["id"])
        if (token["token"].startswith("##")) or (
            token["token"] in ["[CLS]", "[SEP]", "[PAD]"]
        ):
            label_ids.append(-100)
            ner_tags.append(-100)
        else:
            label_ids.append(label2id[token["bio_label"]])
            ner_tags.append(token["bio_label"])

    return {
        "sentence_id": sentence_id,
        "tokens": tokens_list,
        "token_ids": token_ids,
        "ner_tags": ner_tags,
        "label_ids": label_ids,
    }


def get_BIO_tagged_data(dataset, tokenizer):
    results = []
    for sentence_id in dataset:
        result = tokens_to_list(
            sentence_id, generate_labeled_tokens(dataset[sentence_id], tokenizer)
        )
        results.append(result)
    return results


def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    dataset = get_dataset(args.file_path)
    results = get_BIO_tagged_data(dataset, tokenizer)
    with open(args.output_path, "w", encoding="utf-8") as json_file:
        json.dump(results, json_file, ensure_ascii=False, indent=2)
    print(f"Saved to {args.output_path}\nNumber of sentences: {len(results)}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--file_path", type=str, default="./model_training/data/train_data.json"
    )
    parser.add_argument(
        "--output_path", type=str, default="./model_training/data/train_data_BIO.json"
    )
    parser.add_argument("--model", type=str, default="distilbert-base-uncased")
    args = parser.parse_args()
    main(args)
