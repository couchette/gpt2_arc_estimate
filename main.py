import json
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from utils import unzip_file


# Load the pre-trained GPT-2 model and the related tokenizer
model_name_or_path = "gpt2"  # You can choose from "gpt2", "gpt2-medium", "gpt2-large", or "gpt2-xl"
model = GPT2LMHeadModel.from_pretrained(model_name_or_path)
tokenizer = GPT2Tokenizer.from_pretrained(model_name_or_path, padding_side='left')  # Set padding_side to 'left'
tokenizer.pad_token = tokenizer.eos_token  # Set pad_token_id to eos_token_id

unzip_file("data/ARC-V1-Feb2018.zip", "data/ARC")

# Load the ARC dataset
with open('data/ARC/ARC-V1-Feb2018-2/ARC-Easy/ARC-Easy-Test.jsonl', 'r', encoding='utf-8') as file:  # Replace with your actual file path
    arc_data = list(map(json.loads, file))

# Define a function to generate an answer from the model
def generate_answer(question):
    # Encode the question into a format that the model can understand
    inputs = tokenizer.encode(question + tokenizer.eos_token, return_tensors='pt')
    # Generate an attention mask
    attention_mask = torch.tensor([1] * len(inputs[0])).unsqueeze(0)  # Add an extra dimension
    # Generate an answer from the model
    output = model.generate(inputs, attention_mask=attention_mask, max_length=100, do_sample=True)
    # Decode the answer into a human-readable format
    answer = tokenizer.decode(output[:, inputs.shape[-1]:][0], skip_special_tokens=True)
    return answer

# Evaluate the model on the ARC dataset
correct = 0
total = 0
for item in arc_data:
    question = item['question']['stem']
    model_answer = generate_answer(question)
    correct_answers = [choice['text'] for choice in item['question']['choices'] if choice['label'] == item['answerKey']]
    
    print(\
'''
==========question======================
{}
--------model answer----------
{}
------correct answer----------
{}
'''.format(question, model_answer, correct_answers))
    if model_answer in correct_answers:
        correct += 1
    total += 1

    # Print the accuracy of the model on the ARC dataset
    print("Accuracy({}/{}): {}".format(correct, total, correct / total))
