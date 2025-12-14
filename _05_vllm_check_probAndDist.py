
import os
import gc
import pandas as pd

import pickle
from collections import Counter
from tqdm import tqdm
import numpy as np
import random

import json
import ast
from sklearn.metrics import *


import openai
from openai import OpenAI
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)

# Load your API key from an environment variable or secret management service
openai.api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI()

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def completion_with_backoff(**kwargs):
    return client.chat.completions.create(**kwargs)

import tiktoken
encoding = tiktoken.encoding_for_model("gpt-4o")

def slice_text_list(text_list, thr=8000):
    cnt = 0
    final_slice, sub_slice = [], []
    for text in text_list:
        encoded = encoding.encode(text)
        if cnt + len(encoded) < thr:
            sub_slice.append(text)
            cnt += len(encoded)
        else:
            final_slice.append(sub_slice)
            sub_slice = [text]
            cnt = len(encoded)
    final_slice.append(sub_slice)
    return final_slice

def normalize_l2(x):
    x = np.array(x)
    if x.ndim == 1:
        norm = np.linalg.norm(x)
        if norm == 0:
            return x
        return x / norm
    else:
        norm = np.linalg.norm(x, 2, axis=1, keepdims=True)
        return np.where(norm == 0, x, x / norm)

def process_dimensions(input_text):
    text_list = input_text.split('\n')
    return_list = []
    for text in text_list:
        if text.strip() == '':
            continue
        text = text.replace('[Answer]','').replace('**','').replace('- ','').strip()
        if text[0].isdigit():
            text = text[1:]
        if text[0]=='.':
            text = text[1:]
        return_list.append(text.strip())
    return '\n'.join(return_list)

def get_embeddings(all_texts, emb_model="text-embedding-3-large"):
    all_texts_sliced = slice_text_list(all_texts)
    all_embeddings = []
    for texts in tqdm(all_texts_sliced):
        results = []
        prompt_texts = [text.replace('\n',' ') for text in texts]

        response = client.embeddings.create(input=prompt_texts,model=emb_model)
        
        all_embeddings += [item.embedding for item in response.data]
    return all_embeddings

from sklearn import metrics

def get_pairwise_dists(X, Y, emb_dim=256, thr=100000):
    all_dists = []
    for i in tqdm(range(len(X))):
        _dists = metrics.pairwise_distances(np.reshape(X[i],(1,emb_dim)),np.array(Y))[0]
        all_dists.append(_dists[:thr])
    return all_dists

with open(f"{os.getcwd()}/data/author_to_subID_to_inst.pkl", "rb") as f:
    author_to_subID_to_inst = pickle.load(f)

with open(f"{os.getcwd()}/vllm_inference/prob_per_different_evidence_withGold_llama-8b.pkl", "rb") as f:
    _dict = pickle.load(f)

all_IDs = _dict["ID"]
all_seeds = _dict["Seed"]
all_evidence = _dict["Evidence"]
all_probs1 = _dict["prob1"]
all_probs0 = _dict["prob0"]

ID_to_seed_to_predProb = {}
for _ID, _seed, _evidence, _prob1, _prob0 in zip(all_IDs, all_seeds, all_evidence, all_probs1, all_probs0):
    if _ID not in ID_to_seed_to_predProb:
        ID_to_seed_to_predProb[_ID] = {}
    
    _subID, _author = _ID.split("###")[0], _ID.split("###")[1]
    _gold = int(author_to_subID_to_inst[_author][_subID]["judgment"])
    _probs = [_prob0, _prob1]

    ID_to_seed_to_predProb[_ID][_seed] = _probs[_gold]

larger_prob_cnt, smaller_prob_cnt = 0, 0
for _ID in ID_to_seed_to_predProb:
    closest_prob = ID_to_seed_to_predProb[_ID][-1]

    for i in range(10):
        curr_prob = ID_to_seed_to_predProb[_ID][i]
        if curr_prob > closest_prob:
            larger_prob_cnt += 1
        else:
            smaller_prob_cnt += 1

print(f"# case Gold prob > Random prob: {smaller_prob_cnt} ({smaller_prob_cnt/(larger_prob_cnt+smaller_prob_cnt)*100:.2f}%)")

all_newIDs, all_values, all_comments, all_situations = [],[],[],[]
all_values_ref, all_comments_ref, all_situations_ref = [],[],[]
        
for _ID, _seed, _evidence in zip(all_IDs, all_seeds, all_evidence):
    _subID, _author = _ID.split("###")[0], _ID.split("###")[1]

    all_newIDs.append(_ID+"_"+str(_seed))
    all_values.append("\n".join([author_to_subID_to_inst[_author][elem]["abst_value"] for elem in _evidence]))
    all_comments.append("\n".join([author_to_subID_to_inst[_author][elem]["comment"] for elem in _evidence]))
    all_situations.append("\n".join([author_to_subID_to_inst[_author][elem]["selftext"] for elem in _evidence]))

    all_values_ref.append(author_to_subID_to_inst[_author][_subID]["abst_value"])
    all_comments_ref.append(author_to_subID_to_inst[_author][_subID]["comment"])
    all_situations_ref.append(author_to_subID_to_inst[_author][_subID]["selftext"])

values_reduced_emb = [normalize_l2(emb[:256]) for emb in get_embeddings(all_values)]
# comments_reduced_emb = [normalize_l2(emb[:256]) for emb in get_embeddings(all_comments)]
# situations_reduced_emb = [normalize_l2(emb[:256]) for emb in get_embeddings(all_situations)]

values_ref_reduced_emb = [normalize_l2(emb[:256]) for emb in get_embeddings(all_values_ref)]
# comments_ref_reduced_emb = [normalize_l2(emb[:256]) for emb in get_embeddings(all_comments_ref)]
# situations_ref_reduced_emb = [normalize_l2(emb[:256]) for emb in get_embeddings(all_situations_ref)]

print("Computing distance")
values_dist = get_pairwise_dists(values_reduced_emb, values_ref_reduced_emb)

print("Writing files")
with open(f"{os.getcwd()}/outputs/random_evidence_values_reduced_emb.pkl", "wb") as f:
    pickle.dump(values_reduced_emb, f)
with open(f"{os.getcwd()}/outputs/random_evidence_values_ref_reduced_emb.pkl", "wb") as f:
    pickle.dump(values_ref_reduced_emb, f)
with open(f"{os.getcwd()}/outputs/random_evidence_values_to_values_ref_dist.pkl", "wb") as f:
    pickle.dump(values_dist, f)


true_probs, alt_probs = [], []
situ_rel_dists, val_rel_dists, comm_rel_dists = [], [], []
        
for _ID, _seed, _evidence, _prob1, _prob0 in zip(all_IDs, all_seeds, all_evidence, all_probs1, all_probs0):
    if _ID not in ID_to_seed_to_predProb:
        ID_to_seed_to_predProb[_ID] = {}
    
    _subID, _author = _ID.split("###")[0], _ID.split("###")[1]
    _gold = int(author_to_subID_to_inst[_author][_subID]["judgment"])
    _probs = [_prob0, _prob1]

    true_probs.append(_probs[_gold])
    alt_probs.append(_probs[1-_gold])

    new_ID = _ID+"_"+str(_seed)
    _idx = all_newIDs.index(new_ID)

    val_rel_dists.append(values_dist[_idx][_idx])
    # situ_rel_dists.append(situations_dist[_idx][_idx])
    # comm_rel_dists.append(comments_dist[_idx][_idx])

df_prob1 = pd.DataFrame({
    'Distance': val_rel_dists,
    'Probability': true_probs,
    'Type': 'True Prob'
})

df_prob2 = pd.DataFrame({
    'Distance': val_rel_dists,
    'Probability': alt_probs,
    'Type': 'Alt Prob'
})

# Concatenate the two DataFrames
df_combined = pd.concat([df_prob1, df_prob2])
print("Writing dataframe")
df_combined.to_csv(f"{os.getcwd()}/outputs/value_rel_dists_and_probabilities.csv", index=False)

# # 3. Create the Scatterplot using Seaborn
# plt.figure(figsize=(10, 6)) # Set the figure size for better readability

# # Use sns.scatterplot to visualize the data
# # 'x' specifies the column for the x-axis ('Distance')
# # 'y' specifies the column for the y-axis ('Probability')
# # 'hue' distinguishes between 'Probability 1' and 'Probability 2' by color
# sns.scatterplot(
#     data=df_combined,
#     x='Distance',
#     y='Probability',
#     hue='Type',       # Different colors for prob1 and prob2
#     s=100,            # Marker size
#     alpha=0.7,        # Transparency of markers
#     edgecolor='w',    # White edge around markers for better visibility
#     linewidth=0.5     # Line width of the marker edge
# )

# # 4. Customize the Plot
# plt.title('Probability Values vs. Distance', fontsize=16) # Title of the plot
# plt.xlabel('Distance', fontsize=14)                      # X-axis label
# plt.ylabel('Probability (0 to 1)', fontsize=14)          # Y-axis label

# # Set y-axis limits from 0 to 1 as requested
# plt.ylim(0, 1)

# # Add a grid for easier reading of values
# plt.grid(True, linestyle='--', alpha=0.6)

# # Add a legend to identify 'Probability 1' and 'Probability 2'
# plt.legend(title='Probability Type')

# # Display the plot
# plt.show()