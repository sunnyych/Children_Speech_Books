import argparse
import json
import random
from openai import OpenAI
import json
import os
import asyncio
import pandas as pd
from azure import AsyncAzureOpenAI  # Import the Azure OpenAI client
from scipy.spatial import distance
import ast
from tqdm import tqdm
# Initialize the OpenAI API client
# api_key = ''
# client = OpenAI(api_key=api_key)


# Apply nest_asyncio patch for compatibility with interactive environments
import nest_asyncio
nest_asyncio.apply()

# Retrieve the API key from the environment variable
api_key = os.getenv("AZURE_OPENAI_API_KEY")

# Check if the API key was retrieved successfully
if not api_key:
    raise ValueError("API key is not set. Please set the AZURE_OPENAI_API_KEY environment variable.")

# Initialize the Azure OpenAI API client
azure_endpoint = ""  # Your Azure endpoint URL
api_version = ""  # The API version to use

client = AsyncAzureOpenAI(
    azure_endpoint=azure_endpoint,
    api_version=api_version,
    api_key=api_key
)

# Asynchronous function to generate embeddings for a single sentence
async def generate_embedding(sentence):
    response = await client.embeddings.create(
        model="text-embedding-ada-002",  # Use the appropriate model for generating embeddings
        input=[sentence]  # Input should be a list of strings
    )
    embedding = response.data[0].embedding  # Extract the embedding
    return embedding

# Wrapper function to generate embeddings for a list of sentences
def generate_embeddings(sentences_list):
    embeddings = []
    loop = asyncio.get_event_loop()
    for sentence in tqdm(sentences_list, desc="Generating embeddings", unit="sentence"):
        embedding = loop.run_until_complete(generate_embedding(sentence))
        embeddings.append({"sentence": sentence, "embedding": embedding})
    return embeddings


# Function to create prompts based on the category
def create_prompt(category, sentence, similar_pairs):
    examples = "\n".join(
        [f"{i+1}. \"{pair['item']}\" -> {pair['response']}" for i, pair in enumerate(similar_pairs)]
    )
    if category == "generics":
        return (
        "Generics can be categorized into the following five categories: "
        "Behavioral (e.g., \"Fish swim\",  \"Cars park\", \"Sugar melts\"), Biological (e.g., \"Cats have fur\", \"Sharks have pointy teeth\", \"Zebras have long necks\"), "
        "Social (e.g., \"Zebras share food with group members\", \"Bees protect the queen\", \"Wolves travel in packs\"), Teleological (e.g., \"Bees make honey\", \"Cars drive\", \"Clouds provide rain\") and Perceptual (e.g., \"Whales are big\", \"Balls are round\", \"Snow is cold\"). "
        "Behavioral generics are about behavior or what things do; Biological generics mention biological features and characteristics; "
        "Social generics refer to social behaviors or behaviors of groups; Teleological generics are about functions, purposes, or what things are for; and Perceptual generics are about what things look like, or what they feel, sound, or taste like. "
        "For the following sentence, determine if it is a Behavioral, Biological, Social, Teleological, or Perceptual, or Other. "
        "Just return the one-word category name as the output.\n"
        f"Sentence: \"{sentence}\"\n\n"
        "Examples:\n"
        "1. \"A wave must be at least 20 feet (6 m) tall to be considered a 'big' wave.\" -> Perceptual\n"
        "2. \"A 20-gallon (76-1) glass tank is the right size for two or three leopard geckos.\" -> Perceptual\n"
        "3. \"Blue hair is fun to brush and comb.\" -> Teleological\n"
        "4. \"Little snow people, like her, eat nothing but icicles.\" -> Behavioral\n"
        "5. \"Most animals need a mate to have babies.\" -> Teleological\n"
        "6. \"Sharks even allow Pilot fish to clean their teeth!\" -> Behavioral\n"
        "7. \"To keep enemies away, monarchs have built-in protections.\" -> Behavioral\n"
        "8. \"Women are still sometimes paid less than men for the same work.\" -> Social\n"
        "9. \"All birds have wings.\" -> Biological\n"
        "10. \"Humpback Whales don't have teeth.\" -> Biological\n"
        "11. \"What friends do..\" -> Other\n"
        "12. \"Gravity is everywhere.\" -> Other\n"
        f"Sentence: \"{sentence}\"\n\n"
        "More examples:\n"
        f"{examples}\n"
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


# Function to parse embedding strings into lists of floats
def parse_embedding(embedding_str):
    try:
        # Attempt to parse the embedding string into a list of floats
        return ast.literal_eval(embedding_str)
    except:
        # If ast.literal_eval fails, attempt to convert the string manually
        return [float(x.strip()) for x in embedding_str.strip('[]').split(',')]

# Function to classify a new sentence by finding the closest embedding
async def classify_with_embeddings(new_text, new_embedding, search_df):
    # Find the top 5 closest matches using cosine similarity on "item" embeddings
    search_df['similarity'] = search_df['item_embedding'].apply(lambda x: 1 - distance.cosine(x, new_embedding))
    top_matches = search_df.nlargest(5, 'similarity')
    
    # Prepare input-output pairs for the prompt
    similar_pairs = top_matches[['item', 'response']].to_dict('records')
    
    # Create the prompt with the top 5 similar pairs
    prompt = create_prompt(new_text, similar_pairs)
    
    # Generate the final response using the model
    response = await client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "Generate an appropriate response based on the provided context."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=10  # Adjust as needed for response length
    )
    
    final_response = response.choices[0].message.content.strip()
    print(f"New Text: \"{new_text}\" - Generated Response: {final_response}")
    
    return {"sentence": new_text, "classification": final_response, "context": similar_pairs}

# Classify each sentence and save the results to a JSON file
async def classify_sentences(sentences_df, search_csv_path):
    search_df = pd.read_csv(search_csv_path)
    search_df['item_embedding'] = search_df['item_embedding'].apply(parse_embedding)
    
    results = []
    for _, row in tqdm(sentences_df.iterrows(), total=len(sentences_df), desc="Classifying sentences"):
        sentence = row['sentence']
        embedding = parse_embedding(row['embedding'])
        result = await classify_with_embeddings(sentence, embedding, search_df)
        results.append(result)
    
    # Save results to JSON
    output_file = 'OUTPUT.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Classification results saved to {output_file}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classify sentences based on category and data type.")
    parser.add_argument("category", choices=["generics", "habituals"], help="Select a category (generics or habituals).")
    parser.add_argument("data_type", choices=["books", "speech"], help="Select the type of data (books or speech).")

    args = parser.parse_args()

    # Load data based on user input
    data = load_data(args.data_type)

    
    # Generate embeddings for the given list of sentences
    embeddings = generate_embeddings(data)

    # Convert the embeddings to a DataFrame
    df = pd.DataFrame(embeddings)
    search_csv = 'data/features_classification_data/embeddings/embeddings.csv'
    
    # # Ensure embeddings are parsed correctly
    # sentences_df['item_embedding'] = sentences_df['embedding'].apply(parse_embedding)
    
    # Classify sentences
    classify_sentences(df, search_csv)


    output_file = 'classified_sentences.json'

    with open(output_file, 'w') as f:
        json.dump(classified_sentences, f, indent=4)

    print(f"Classification results saved to {output_file}")

 
