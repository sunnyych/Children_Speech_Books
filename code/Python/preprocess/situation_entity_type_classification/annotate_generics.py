import json
# pass in a dict of clauses
with open("data/extracted_clauses/extracted_clauses.json", "r") as f:
    clauses_parents = json.load(f)

# load fine-tuned model

from transformers import BertTokenizer, BertForSequenceClassification

# Define the path where the fine-tuned model is saved
model_save_path = "/code/Python/situation_entity_type_classification/trained_model"

# Load the tokenizer
tokenizer = BertTokenizer.from_pretrained(model_save_path)

# Load the fine-tuned model
model = BertForSequenceClassification.from_pretrained(model_save_path)

import torch
from tqdm import tqdm  # Import tqdm for the progress bar

# Ensure the model is on the correct device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.eval()

def classify_clauses(clauses, model, tokenizer, device):
    results = []
    # Wrap the loop with tqdm for a progress bar
    for clause in tqdm(clauses, desc="Classifying clauses"):
        inputs = tokenizer(clause['sentence'], return_tensors='pt', padding=True, truncation=True, max_length=128)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=1)
            clause['category'] = predictions.item()
        results.append(clause)
    return results

# Classify clauses
annotated_clauses = classify_clauses(clauses_parents, model, tokenizer, device)

import json

# Save annotated clauses to a JSON file
with open('annotated_clauses.json', 'w') as f:
    json.dump(annotated_clauses, f, indent=4)
