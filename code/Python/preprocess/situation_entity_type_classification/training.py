import os
import re
import numpy as np
import torch
import json
import pickle as cPickle
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold
from collections import Counter
from bs4 import BeautifulSoup

# Initialize the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Load the pre-trained BERT model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=8)

# Load document lists
train_list = []
test_list = []

file_path = ""  # Include the file path for "train_test_split"
with open(file_path, 'r') as f:
    for line in f.readlines():
        filename, genre, index, train_test = line.split()
        filename = filename[:-4]
        if train_test == 'train':
            train_list.append(filename)
        if train_test == 'test':
            test_list.append(filename)

# Load stored pos/ner parsing sentence
file_path1 = ""  # Path to "masc_sentence_pos_ner_dict.pkl"
file_path2 = ""  # Path to "explicit_connective.txt"

with open(file_path1, 'rb') as f:
    sentence_pos_ner_dict = cPickle.load(f)

connective_list = []
with open(file_path2, 'r') as f:
    for line in f.readlines():
        connective_list.append(line.strip())
connective_list = tuple(connective_list)

entity_type_list = ['STATE', 'EVENT', 'REPORT', 'GENERIC_SENTENCE', 'GENERALIZING_SENTENCE', 'QUESTION', 'IMPERATIVE', "CANNOT_DECIDE"]

def process_entity_type_label(entity_type):
    return entity_type_list.index(entity_type)

# Paths to data
path1 = ""  # Path to "annotations_xml"
path2 = ""  # Path to "raw_text"

class ClauseDataset(Dataset):
    def __init__(self, doc_list, tokenizer, max_length=128):
        self.clauses = []
        self.labels = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.process_docs(doc_list)
        print(f"Total clauses processed: {len(self.clauses)}")
        print(f"Total labels processed: {len(self.labels)}")

    def process_docs(self, doc_list):
        for doc_name in doc_list:
            doc_path = os.path.join(path1, doc_name + '.xml')
            raw_doc_path = os.path.join(path2, doc_name + '.txt')
            doc_clause_list = self.process_doc(doc_path)
            paras_clause_list = self.process_paragraph(doc_clause_list, raw_doc_path)
            for para_clause_list in paras_clause_list:
                for clause in para_clause_list:
                    clause_text, entity_type = clause[0], clause[1]
                    if entity_type in entity_type_list:
                        self.clauses.append(clause_text)
                        self.labels.append(process_entity_type_label(entity_type))
            print(f"Processed {len(self.clauses)} clauses so far.")

    def process_doc(self, doc_path):
        with open(doc_path, 'r') as doc:
            xml = BeautifulSoup(doc, "lxml-xml")  # Use lxml parser for XML
            clause_list = []
            for clause in xml.find_all('segment'):
                end = int(clause.attrs['end'])
                clause_text = clause.find('text').string
                label = 'CANNOT_DECIDE'
                annotation = clause.find('annotation', attrs={"annotator": "gold"})
                if annotation.has_attr('seType'):
                    label = annotation.attrs['seType']
                clause_list.append((clause_text, label, end))
            return clause_list

    def process_paragraph(self, doc_clause_list, raw_doc_path):
        with open(raw_doc_path, 'r') as raw_doc:
            raw_para_boundary_list = [m.start() + 1 for m in re.finditer('\n\n', raw_doc.read())]
            raw_para_boundary_list += [float('inf')]
            paras_clause_list = []

            index = 0
            for raw_para_boundary in raw_para_boundary_list:
                para_clause_list = []
                while index < len(doc_clause_list):
                    end_index = doc_clause_list[index][2]
                    if end_index <= raw_para_boundary:
                        para_clause_list.append(doc_clause_list[index])
                        index += 1
                    else:
                        break
                if len(para_clause_list) > 0:
                    paras_clause_list.append(para_clause_list)
            return paras_clause_list

    def __len__(self):
        return len(self.clauses)

    def __getitem__(self, idx):
        clause = self.clauses[idx]
        label = self.labels[idx]
        encoding = self.tokenizer.encode_plus(
            clause,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

train_dataset = ClauseDataset(train_list, tokenizer)
test_dataset = ClauseDataset(test_list, tokenizer)

train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False)

def calculate_label_distribution(dataset, entity_type_list):
    label_counts = np.zeros(len(entity_type_list), dtype=int)
    for item in dataset:
        label = item['labels'].item()  # Extract the label
        label_counts[label] += 1
    return label_counts

# Calculate label distribution for training dataset
train_label_counts = calculate_label_distribution(train_dataset, entity_type_list)
test_label_counts = calculate_label_distribution(test_dataset, entity_type_list)

# Print label distribution
print("Training Set Label Distribution:")
for i, count in enumerate(train_label_counts):
    print(f"{entity_type_list[i]}: {count}")

print("\nTest Set Label Distribution:")
for i, count in enumerate(test_label_counts):
    print(f"{entity_type_list[i]}: {count}")

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    eval_strategy='epoch'
)

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    accuracy = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics
)

# Train the model
trainer.train()

# Save the trained model
model_save_path = "trained_model"
trainer.save_model(model_save_path)
tokenizer.save_pretrained(model_save_path)

# Save the training arguments
training_args_dict = training_args.to_dict()
with open(os.path.join(model_save_path, 'training_args.json'), 'w') as f:
    json.dump(training_args_dict, f)

# Evaluate the model
results = trainer.evaluate(eval_dataset=test_dataset)
print(results)

# Calculate label distributions in the training and test sets
train_labels = [label for label in train_dataset.labels]
test_labels = [label for label in test_dataset.labels]

train_label_distribution = Counter(train_labels)
test_label_distribution = Counter(test_labels)

print(f"Training label distribution: {train_label_distribution}")
print(f"Test label distribution: {test_label_distribution}")

def cross_validate_model(model, dataset, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits)
    metrics_list = []

    for train_index, val_index in skf.split(dataset.clauses, dataset.labels):
        train_subset = torch.utils.data.Subset(dataset, train_index)
        val_subset = torch.utils.data.Subset(dataset, val_index)

        train_dataloader = DataLoader(train_subset, batch_size=8, shuffle=True)
        val_dataloader = DataLoader(val_subset, batch_size=8, shuffle=False)

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_subset,
            eval_dataset=val_subset,
            compute_metrics=compute_metrics
        )

        trainer.train()
        metrics = trainer.evaluate(eval_dataset=val_subset)
        metrics_list.append(metrics)

    avg_metrics = {
        'eval_loss': np.mean([m['eval_loss'] for m in metrics_list]),
        'eval_accuracy': np.mean([m['eval_accuracy'] for m in metrics_list]),
        'eval_precision': np.mean([m['eval_precision'] for m in metrics_list]),
        'eval_recall': np.mean([m['eval_recall'] for m in metrics_list]),
        'eval_f1': np.mean([m['eval_f1'] for m in metrics_list])
    }

    return avg_metrics

# Perform cross-validation
cross_val_metrics = cross_validate_model(model, train_dataset)
print(f"Cross-validation metrics: {cross_val_metrics}")
