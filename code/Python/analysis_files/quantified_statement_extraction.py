import argparse
import json
import random
from openai import OpenAI
import json
import re
from tqdm import tqdm
from nltk.corpus import stopwords




# Function to load data based on type
def load_data(category, data_type):
    if data_type == "books":
        print("Loading data for books...")
        # Load the books data
        with open('MERGED_DATA_BOOKS.json', 'r') as file:
            data = json.load(file)
        preprocessed_data = preprocess_book_data(category, data)
        # Add your filter function here if needed
        return preprocess_book_data
    elif data_type == "speech":
        print("Loading data for speech...")
        # Load the speech data
        with open('MERGED_DATA_SPEECH.json', 'r') as file:
            data = json.load(file)
        # Preprocess data if needed
        preprocessed_data = preprocess_speech_data(category, data)
        return preprocessed_data
    else:
        raise ValueError(f"Unknown data type: {data_type}")

# Placeholder function for speech data preprocessing
def preprocess_speech_data(category, data):
    print("Preprocessing speech data...")
    if category == "generics":
        number = 3
    elif category == "habituals":
        number = 4
    filtered_books = []
    for book in data:
        for sentence in book:
            if sentence['category'] == number:
                filtered_books.append(sentence['sentence'])
    
    return filtered_books


def preprocess_book_data(category, data):
    print("Preprocessing speech data...")
    if category == "generics":
        number = 3
    elif category == "habituals":
        number = 4
    filtered_generics = []
    for sentence in data:
        if sentence['category'] == number:
            filtered_generics.append(sentence['sentence'])
    return filtered_generics



excluded_pronouns = {
    'he', 'you', 'we', 'she', 'i', 'they', 'him', 'her', 'them', 'his', 'hers', 'theirs', 'it', 'way', 'thing'
}
excluded_pronouns.update(set(stopwords.words('english')))

# Augmented lists of universal and existential quantifiers
universal_quantifiers = [
    "all", "every", "each", "any", "no", "none", "never", "always",
    "everybody", "everyone", "everything", "everywhere", "no one",
    "nobody", "nothing", "neither", "every single", "without exception",
    "entire", "whole", "everywhere", "whenever"
]

existential_quantifiers = [
    "some", "a", "an", "one", "several", "many", "few", "a few",
    "a couple of", "at least one", "a number of", "a lot of",
    "a majority of", "a minority of", "certain", "most", "almost all",
    "more than half", "half of", "less than half", "two-thirds",
    "three-quarters", "dozens of", "hundreds of", "thousands of",
    "millions of", "billions of", "about", "approximately", "various",
    "plenty of", "somebody", "someone", "something", "somewhere",
    "multiple", "varied", "part of", "enough", "much",
    "a portion of", "particular", "certain individuals", "occasional"
]

def categorize_sentence_by_subject(sentence, subject, universal_pattern, existential_pattern):
    # Normalize the sentence and subject to lowercase for easier matching
    sentence = sentence.lower()
    subject = subject.lower()

    # Check if any universal quantifier appears directly before the subject
    if re.search(universal_pattern, sentence):
        return "Universal Quantifier"

    # Check if any existential quantifier appears directly before the subject
    if re.search(existential_pattern, sentence):
        return "Existential Quantifier"

    # If no quantifiers are found directly before the subject
    return "Neither"

def process_data(data, data_type, quantifier_type, output_file):
    # Determine which quantifiers to use
    quantifiers = universal_quantifiers if quantifier_type == "universal" else existential_quantifiers

    # Process the data based on the data type
    for item in tqdm(data, desc=f"Processing {data_type}"):
        if data_type == "books":
            for sentence_data in item:
                process_sentence_data(sentence_data, quantifiers)
        elif data_type == "speech":
            process_sentence_data(item, quantifiers)

    # Save the updated data to a new JSON file
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=4)

def process_sentence_data(sentence_data, quantifiers):
    sentence = sentence_data.get('sentence', "")
    subjects = sentence_data.get('subjects', [])
    if subjects:
        subject_text = subjects[0].get('subject', "")
        if subject_text in excluded_pronouns:
            sentence_data['quantified_classification'] = "Neither"
        else:
            # Compile the regular expressions using the current subject
            universal_pattern = re.compile(r'\b(?:' + '|'.join(map(re.escape, universal_quantifiers)) + r')\b\s+\b' + re.escape(subject_text) + r'\b')
            existential_pattern = re.compile(r'\b(?:' + '|'.join(map(re.escape, existential_quantifiers)) + r')\b\s+\b' + re.escape(subject_text) + r'\b')

            # Categorize the sentence based on the subject
            classification = categorize_sentence_by_subject(sentence, subject_text, universal_pattern, existential_pattern)
            # Add the classification to the sentence data
            sentence_data['quantified_classification'] = classification
    else:
        # If no subject is present, classify as "Neither"
        sentence_data['quantified_classification'] = "Neither"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classify sentences based on category and data type.")
    parser.add_argument("data_type", choices=["books", "speech"], help="Select the type of data (books or speech).")
    parser.add_argument("quantifier_type", choices=["universal", "existential"], help="Select the type of quantifier (universal or existential).")
    args = parser.parse_args()
    data = load_data(args.data_type)
    output_file = 'output.json'  # Replace with actual output file path
    process_data(data, args.data_type, args.quantifier_type, output_file)

 
