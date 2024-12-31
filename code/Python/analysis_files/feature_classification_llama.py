import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import pandas as pd
from tqdm import tqdm
import random
import json
import os

def main(model_name, feature_type):
    # Load Llama 8B model and tokenizer
    print(f"Loading model: {model_name}")
    cache_dir = "/juice2/scr2/syu03"
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Model loaded on {device}")
    # load data
    with open('MERGED_DATA_BOOKS.json', 'r') as file:
        generics_data = json.load(file)
    generics = []
    if feature_type == "generic":
        number = 3
    elif feature_type == "habitual":
        number = 4
    for book in generics_data:
        for sentence in book:
            if sentence['category'] == number:
                generics.append(sentence['sentence'])

    # Run inference on the test dataset
    prompts = []
    def create_prompt(sentence, feature_type):
        if feature_type == "generic":
            return (
                "Generics can be categorized into the following five categories: "
                "Behavioral (e.g., \"Fish swim\",  \"Cars park\", \"Sugar melts\"), Biological (e.g., \"Cats have fur\", \"Sharks have pointy teeth\", \"Zebras have long necks\"), "
                "Social (e.g., \"Zebras share food with group members\", \"Bees protect the queen\", \"Wolves travel in packs\"), Teleological (e.g., \"Bees make honey\", \"Cars drive\", \"Clouds provide rain\") and Perceptual (e.g., \"Whales are big\", \"Balls are round\", \"Snow is cold\"). "
                "Behavioral generics are about behavior or what things do; Biological generics mention biological features and characteristics; "
                "Social generics refer to social behaviors or behaviors or groups; Teleological generics are about functions, purposes, or what things are for; and Perceptual generics are about what things look like, or what they feel, sound or taste like"
                "For the following sentence, determine if it is a Behavioral, Biological, Social, Teleological or Perceptual generic, "
                "Return 0 for None, 1 for Behavioral, 2 for Biological, 3 for Social, 4 for Teleological, and 5 for Perceptual. Just return the number, no explanation. Perform multi-class classification, and use commas to separate the numbers. Below are some examples:\n"
                "Examples:\n"
                "1. \"A wave must be at least 20 feet (6 m) tall to be considered a 'big' wave.\" -> 5\n"
                "2. \"A 20-gallon (76-1) glass tank is the right size for two or three leopard geckos.\" -> 4,5\n"
                "3. \"Blue hair is fun to brush and comb.\" -> 2,4\n"
                "4. \"Little snow people, like her, eat nothing but icicles.\" -> 1\n"
                "5. \"Most animals need a mate to have babies.\" -> 1,3,4\n"
                "6. \"Sharks even allow Pilot fish to clean their teeth!\" -> 1\n"
                "7. \"To keep enemies away, monarchs have built-in protections.\" -> 1\n"
                "8. \"Women are still sometimes paid less than men for the same work.\" -> 1,3\n"
                "9. \"All birds have wings.\" -> 2\n"
                "10. \"Humpback Whales don't have teeth.\" -> 2\n"
                f"Sentence: \"{sentence}\""
            )
        elif feature_type == "habitual":
            return (
                "Habituals can be categorized into the following five categories: "
                  "Behavioral (e.g., \"This fish is swimming\",  \"John walks to school every day\"), Biological (e.g., \"Some cats have fur\", \"All sharks have pointy teeth\"), "
        "Social (e.g., \"Charlie likes to go out with friends\", \"Bee Bob protects the queen\"), Teleological (e.g., \"This hammer is used to cut the tree\") and Perceptual (e.g., \"This dog is big\", \"Some balls are round\"). "
        "Behavioral sentences are about behavior or what things do; Biological sentences mention biological features and characteristics; "
        "Social sentences refer to social behaviors or behaviors or groups; Teleological sentences are about functions, purposes, or what things are for; and Perceptual sentences are about what things look like, or what they feel, sound or taste like"
                "For the following sentence, determine if it is a Behavioral, Biological, Social, Teleological or Perceptual generic, "
                "Return 0 for None, 1 for Behavioral, 2 for Biological, 3 for Social, 4 for Teleological, and 5 for Perceptual. Just return the number, no explanation. Perform multi-class classification, and use commas to separate the numbers. Below are some examples:\n"
                "Examples:\n"
                 "1. \"Mary often reads a book before bed.\" -> 1\n"
        "2. \"John typically drinks coffee in the morning.\" -> 1\n"
        "3. \"The office staff usually have lunch together at noon.\" -> 1,3\n"
        "4. \"Sarah always checks her emails first thing in the morning.\" -> 1,3\n"
        "5. \"The bee makes honey.\" -> 1,4\n"
        "6. \"The clock is used for reading time.\" -> 1,4\n"
        "7. \"The ball is red.\" -> 5\n"
        "8. \"The sun shines very bright.\" -> 5\n"
        "9. \"The girl has long hair.\" -> 2\n"
        "10. \"James has big hands.\" -> 2\n"
        "11. \"The lady woke up early yesterday.\" -> 1\n"
             
                f"Sentence: \"{sentence}\""
            )

    def classify_sentences(generics):
        results = []
        for sentence in tqdm(generics, desc="Running Inference", unit="prompt"):
            prompt = create_prompt(sentence)
            messages = [{"role": "user", "content": prompt},]
            input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)
            terminators = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]
            outputs = model.generate(
                    input_ids,
                    max_new_tokens=10,
                    eos_token_id=terminators,
                    do_sample=True,
                    temperature=0.6,
                    top_p=0.9,
                )
            response = outputs[0][input_ids.shape[-1]:]
            model_response = tokenizer.decode(response, skip_special_tokens=True)
            results.append({"sentence": sentence, "classification": model_response})
            print(f"Sentence: \"{sentence}\" - Classification: {model_response}")
        return results
    
    classified_sentences = classify_sentences(generics)
    output_file = 'feature_classification.json'

    with open(output_file, 'w') as f:
        json.dump(classified_sentences, f, indent=4)    

    # Save the responses in a new column and write to CSV
    # df["prompt"] = prompts
    # df["llama_response"] = results
    # df.to_csv(f"llama_{prompt_type}_{ablation}_response.csv", index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference with Llama 8B model")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model (e.g., llama-8B)")
    parser.add_argument("--feature_type", type=str, required=True, help="generic or habitual")
    args = parser.parse_args()
    model_name = args.model_name
    feature_type = args.feature_type
    main(model_name, feature_type)









