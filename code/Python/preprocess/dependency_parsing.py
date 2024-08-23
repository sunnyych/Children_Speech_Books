import json
with open("data/dependency_parsing_data/results/annotate_clauses_parents_final.json", "r") as f:
    annotated_clauses = json.load(f)

# for analyzing children's books, use "extracted_clauses_books.json"

import json
import stanza
from tqdm import tqdm

# Initialize the Stanza pipeline
stanza.download('en')
nlp = stanza.Pipeline('en')

# Functions to find ancestors and descendants
def find_ancestors(word, sentence_words):
    ancestors = []
    current = word
    while current.head > 0:  # Head of 0 means the word is the root
        current = sentence_words[current.head - 1]  # Head is 1-based index
        ancestors.append(current.text)
    return ancestors

def find_descendants(word, sentence_words):
    descendants = []
    stack = [word]
    while stack:
        current = stack.pop()
        for w in sentence_words:
            if w.head == current.id:  # If the word points to the current word
                descendants.append(w.text)
                stack.append(w)
    return descendants

# Function to extract subjects and their related words from a clause
def extract_subjects_and_related_words(clause_text):
    doc = nlp(clause_text)
    results = []

    for sentence in doc.sentences:
        for word in sentence.words:
            if word.deprel in ('nsubj', 'nsubjpass'):
                result = {
                    'subject': word.text,
                    'words_pointing_to_subject': find_ancestors(word, sentence.words),
                    'words_subject_points_to': find_descendants(word, sentence.words)
                }
                results.append(result)
    return results

# Function to process a single clause
def process_clause(clause):
    clause_text = clause['sentence']
    clause['subjects'] = extract_subjects_and_related_words(clause_text)
    return clause

# Function to process clauses
def process_clauses(clauses):
    results = []
    for clause in tqdm(clauses, desc="Processing clauses"):
        results.append(process_clause(clause))
    return results


# Process the clauses
processed_clauses = process_clauses(annotated_clauses)

# Save the processed clauses to a JSON file
with open('dependency_parsing.json', 'w') as f:
    json.dump(processed_clauses, f, indent=4)


