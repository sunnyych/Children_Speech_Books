import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset
import json
import evaluate
from transformers import (
    AutoTokenizer,
    DataCollatorForTokenClassification,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
)
from argparse import ArgumentParser

seqeval = evaluate.load("seqeval")

import numpy as np

from torch import cuda

from analysis.configs import change_path_to_root
if __name__ == "__main__":
    change_path_to_root()

DEVICE = "cuda" if cuda.is_available() else "cpu"
print(f"Using {DEVICE}")

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


class NERDataset(Dataset):
    def __init__(self, data, max_len=128):
        self.len = len(data)
        self.data = data
        self.max_len = max_len

    def __getitem__(self, index):
        maxlen = self.max_len
        ids = self.data[index]["token_ids"]
        label_ids = self.data[index]["label_ids"]

        if len(ids) > maxlen:
            # truncate
            ids = ids[:maxlen]
            label_ids = label_ids[:maxlen]

        attn_mask = [1] * len(ids)

        return {
            "input_ids": torch.tensor(ids, dtype=torch.long).to(DEVICE),
            "attention_mask": torch.tensor(attn_mask, dtype=torch.long).to(DEVICE),
            "labels": torch.tensor(label_ids, dtype=torch.long).to(DEVICE),
        }

    def __len__(self):
        return self.len


def get_dataset(file_path):
    training_data = json.load(open(file_path, "r", encoding="utf-8"))
    train_data, test_data = train_test_split(
        training_data, test_size=0.2, random_state=42
    )
    train_dataset = NERDataset(train_data)
    test_dataset = NERDataset(test_data)
    print("train dataset size: ", len(train_dataset))
    print("test dataset size: ", len(test_dataset))
    return train_dataset, test_dataset


def train(model_name, train_dataset, test_dataset, epochs=3, batch_size=16, lr=3e-5):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    def compute_metrics(p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        true_predictions = [
            [id2label[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [id2label[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        results = seqeval.compute(
            predictions=true_predictions, references=true_labels, mode="strict"
        )
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
            "results": results,
        }

    model = AutoModelForTokenClassification.from_pretrained(
        model_name, num_labels=len(id2label), id2label=id2label, label2id=label2id
    ).to(DEVICE)

    training_args = TrainingArguments(
        output_dir="distilbert",
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        weight_decay=0.01,
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    return trainer


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--file_path", type=str, default="./model_training/data/train_data_BIO.json")
    parser.add_argument("--model", type=str, default="distilbert-base-uncased")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--output_path", type=str)
    args = parser.parse_args()
    if not args.output_path:
        args.output_path = args.model.split("-")[0]
        args.output_path = f"./model/{args.output_path}"

    train_dataset, test_dataset = get_dataset(args.file_path)
    trainer = train(
        args.model, train_dataset, test_dataset, args.epochs, args.batch_size, args.lr
    )
    trainer.save_model(args.output_path)
    print(f"Saved model to {args.output_path}")
    print("Done!")
