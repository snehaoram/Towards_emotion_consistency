import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import matplotlib.pyplot as plt
# import seaborn as sns
import pandas as pd
from huggingface_hub import login
import numpy as np
import random
import pickle

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

login(token="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")

# 1. Load Model and Tokenizer
# We use load_in_4bit=True to make it runnable on consumer GPUs.
model_name = "mistralai/Mistral-7B-Instruct-v0.3"
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(model_name, output_attentions=True)


df = pd.read_csv('ExEmo_Mistral.csv')

# df = df[:5]

# def generate_prompt_l1(converse, mo_emo):
#     return f"""You are an helpful assistant. Consider the following conversation between a help-seeker and a supporter. {converse}\nIs the seeker feeling {mo_emo}? Give your answer as 'Yes' or 'No'."""


# def generate_prompt_l2(converse, mo_emo):
#     return f"""You are an helpful assistant. Consider the following conversation between a help-seeker and a supporter. {converse}\nLooks like the seeker feeling {mo_emo}? Do you agree? Give a very short response."""


# def generate_prompt_l3(converse, emo, mo_emo):
#     return f"""You are an helpful assistant. Consider the following conversation between a help-seeker and a supporter. {converse}\nWhile the person is {emo}, their words suggest they might be feeling {mo_emo} Do you agree? Give a very short response."""  # speaks about


# def generate_prompt_l4(converse, mo_emo):
#     return f"""You are an helpful assistant. Consider the following conversation between a help-seeker and a supporter. {converse}\nClearly the person is feeling {mo_emo}. Do you agree? Give a very short response."""

finale = []

# 5. NEW: Find and Print Top-K Attended-To Tokens 
def get_top_k_attention(attention_scores_for_token, tokens, k):
        """
        Finds the top K tokens that a specific query token is attending to.
        
        Args:
            attention_scores_for_token (np.array): A 1D array of attention scores from one token to all others.
            tokens (list): The list of all token strings in the sequence.
            k (int): The number of top tokens to return.
            
        Returns:
            list: A list of (token, score) tuples.
        """
        # Get the indices of the top K scores
        top_k_indices = np.argsort(attention_scores_for_token)[-k:][::-1]
        
        top_k_tokens = []
        for i in top_k_indices:
            token = tokens[i]
            score = attention_scores_for_token[i]
            top_k_tokens.append((token, score))
            
        return top_k_tokens
    

for i, row in df.iterrows():
    converse = row['conversation']
    ex_emotion = row['extreme']
    emo = row['Emotion']
    query1 = "Is the speaker feeling" + ex_emotion + "? Give your answer as 'Yes' or 'No'." #"Is the speaker feeling existentially adrift, profoundly unfulfilled, and suffocated by relentless void?"
    query2 = "Is the speaker feeling" + ex_emotion + "? Do you agree? Give a very short response." 
    query3 = "Is the speaker feeling" + ex_emotion + "? Do you agree? Give a very short response." 
    query4 = "Is the speaker feeling" + ex_emotion + "? Do you agree? Give a very short response." 

    # Format the prompt using the official Mistral instruction template
    prompt1 = f"Consider a conversation between a help-seeker and a supporter: Conversation: {converse} \n\n Query: {query1}"
    prompt2 = f"Consider a conversation between a help-seeker and a supporter: Conversation: {converse} \n\n Query: {query2}"
    prompt3 = f"Consider a conversation between a help-seeker and a supporter: Conversation: {converse} \n\n Query: {query3}"
    prompt4 = f"Consider a conversation between a help-seeker and a supporter: Conversation: {converse} \n\n Query: {query4}"

    tokenizer.pad_token_id = tokenizer.eos_token_id
    # 3. Tokenize the Input
    inputs2 = tokenizer(prompt2, return_tensors="pt").to(device)  ###########Change these to 2,3,4 for other query levels
    token_strings2 = tokenizer.convert_ids_to_tokens(inputs2["input_ids"][0])    ###########Change these to 2,3,4 for other query levels


    model.to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs2,
            max_new_tokens=1, # We only care about the attention on the prompt
            output_attentions=True,
            return_dict_in_generate=True
        )


    # The attention scores are in `outputs.attentions`.
    # It's a tuple where each element corresponds to a layer.
    # Structure: (layer, batch, head, sequence_length, sequence_length)
    attentions = outputs.attentions[0] # Get attentions for the first generated token


    # --- Parameters for Visualization ---
    layer_to_visualize = 31  # Mistral-7B has 32 layers (0-31)
    head_to_visualize = 31   # Mistral-7B has 32 heads (0-31)


        
    # Extract the attention scores for the specified layer and head
    # We use [0] for the batch size and get the numpy array
    attention_for_head = attentions[layer_to_visualize][0, head_to_visualize, :, :].cpu().numpy()


    query_token_index = -2  #-3 
    top_k = 10   

    # This is the ROW corresponding to our chosen query token
    attention_from_query_token = attention_for_head[query_token_index]

    # Get the top K tokens
    top_attended_tokens = get_top_k_attention(attention_from_query_token, token_strings2, k=top_k)  ############Change it to 2,3,4

    query_token_string = token_strings2[query_token_index]  ############Change it to 2,3,4


    #Additional evaluation
    inputs_q = tokenizer(query2, return_tensors="pt").to(device)   ############Change it to 2,3,4
    token_strings_q = tokenizer.convert_ids_to_tokens(inputs_q["input_ids"][0])

    toks = [item[0] for item in top_attended_tokens]

    finals = []
    for tok in token_strings_q:
        if tok in toks:
            # print(tok)
            finals.append(tok)
    
    finale.append(finals)



print('finale[:5]: ', finale[:5])
filename = 'finale.pkl'

# Open the file in write-binary mode
with open(filename, 'wb') as file:
    # Use pickle.dump() to write the list to the file
    pickle.dump(finale, file)

print(f"List has been successfully saved to {filename}")
