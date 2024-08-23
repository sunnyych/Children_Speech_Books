import argparse
import json
import random
from openai import OpenAI
import json
# Initialize the OpenAI API client
api_key = ''
client = OpenAI(api_key=api_key)

# Function to create prompts based on the category
def create_prompt(category, sentence):
    if category == "generics":
        return (
            "Generics are linguistic expressions that make statements about or refer to kinds, "
            "or that report regularities of events. Non-generic expressions make statements about "
            "particular individuals or specific episodes. For example, sentences 'lions live for 10 years', "
            "'the lion is a predatory cat', and 'a lion may eat up to 30 kg every meal' are generics while "
            "sentences like 'Simba is in danger' and 'John likes to cycle' are not generics. "
            "Based on this definition and the following examples, can you determine if the following sentence is a generic? "
            "Answer '1' if it is and '0' if it is not. Just return the number.\n"
            "\n"
            "Examples:\n"
            "Input: 'a guinea pig doesn't have a tail like that I don't think'\n"
            "Output: 1\n"
            "Input: 'who's feeding the baby'\n"
            "Output: 0\n"
            "Input: 'that's the way you make a chair'\n"
            "Output: 0\n"
            "Input: 'the skin when as a banana is growing it's usually a light green color and turns yellow as it gets riper and ready to eat'\n"
            "Output: 1\n"
            "Input: 'elephants do not eat meat'\n"
            "Output: 1\n"
            "Input: 'if you were working in a bank where everyone wears their outfit'\n"
            "Output: 0\n"
            "Input: 'hyena pups laugh at everything'\n"
            "Output: 1\n"
            "Input: 'so we have a lot of things that have something else in it that shake'\n"
            "Output: 0\n"
            "Input: 'a balloon looks like a circle don't it'\n"
            "Output: 1\n"
            "Input: 'we don't have the right colors for a tree but we can make a green tree'\n"
            "Output: 0\n"
            "Input: 'adult girls are called women'\n"
            "Output: 1\n"
            "Input: 'elephants live and travel in groups called herds'\n"
            "Output: 1\n"
            "Input: 'shark goes in the ocean'\n"
            "Output: 1\n"
            "Input: 'some attract and some repel'\n"
            "Output: 0\n"
            "Input: 'toasted turnips is something to eat but rubber duckies don't say quack'\n"
            "Output: 1\n"
            "Input: 'hyraxes live on the in the rock rocky holes in Africa'\n"
            "Output: 1\n"
            "Input: 'who can cook cookies'\n"
            "Output: 0\n"
            "Input: 'Asian elephants have smoother trunks that end in just one point'\n"
            "Output: 1\n"
            "Input: 'a thief is somebody that steals stuff off you'\n"
            "Output: 1\n"
            "Input: 'a squirrel's face looks kinda like a bunny's face'\n"
            "Output: 1\n"
            "\n"
            f"Sentence: \"{sentence}\""
        )
    elif category == "habituals":
        return (
            "Generalizing sentences are generalizations over events (habituals). The subjects can be specific (such as individuals) but the events are generalized."
            "For example, the sentence 'Mary often feeds the cat' is a generalizing sentence "
            "while 'Mary fed the cat yesterday' is not."
            "Based on this definition and the following examples, determine if the following sentence is generalizing. "
            "Return 1 if it is and 0 if it is not.\n"
            "\n"
            "Examples:\n"
            "Input: 'Brian the little boy that goes to gymnastics'\n"
            "Output: 1\n"
            "Input: 'and we do we do read lots of books'\n"
            "Output: 1\n"
            "Input: 'we stay clear of cats'\n"
            "Output: 1\n"
            "Input: 'we usually cut off the crown the top of it and eat that part every day'\n"
            "Output: 1\n"
            "Input: 'Cinderella woke up early to do chores for her stepmother and stepsisters'\n"
            "Output: 0\n"
            "Input: 'well you eat what you drink see'\n"
            "Output: 0\n"
            "Input: 'you pulling those beads'\n"
            "Output: 0\n"
            "Input: 'I always forget to put the answering machine on'\n"
            "Output: 1\n"
            "Input: 'I eat everything at the movies'\n"
            "Output: 1\n"
            "Input: 'but particularly he uses it to get into the house even when he kind of helps himself in'\n"
            "Output: 0\n"
            "Input: 'on the west coast you usually get mandarins'\n"
            "Output: 1\n"
            "Input: 'mamma makes coffee when she goes to work huh'\n"
            "Output: 1\n"
            "Input: 'the observatory never has monkeys in it'\n"
            "Output: 1\n"
            "\n"
            f"Sentence: \"{sentence}\""
        )
    else:
        raise ValueError(f"Unknown category: {category}")

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




# Function to call the API for each sentence
def classify_sentences(category, sentences):
    results = []
    for sentence in sentences:
        prompt = create_prompt(category, sentence)
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "determine if the sentence is a generic."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1  # Limit the output to 1 token
        )
        # Extract the model's response
        classification = response.choices[0].message.content.strip()
        results.append({"sentence": sentence, "classification": classification})
        print(f"Sentence: \"{sentence}\" - Classification: {classification}")
    return results





if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classify sentences based on category and data type.")
    parser.add_argument("category", choices=["generics", "habituals"], help="Select a category (generics or habituals).")
    parser.add_argument("data_type", choices=["books", "speech"], help="Select the type of data (books or speech).")

    args = parser.parse_args()

    # Load data based on user input
    data = load_data(args.data_type)

    classified_sentences = classify_sentences(args.category, data)

    output_file = 'classified_sentences.json'

    with open(output_file, 'w') as f:
        json.dump(classified_sentences, f, indent=4)

    print(f"Classification results saved to {output_file}")

 
