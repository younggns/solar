import os
import pandas as pd
from tqdm import tqdm
import pickle

from _00_data_loader import get_data_split_author

import numpy as np
import torch
from vllm import LLM, SamplingParams

import tiktoken
encoding = tiktoken.encoding_for_model("gpt-4o")

def truncate_text(text, thr=1024):
    if len(text) == 0:
        return text
    encoded = encoding.encode(text)
    return encoding.decode(encoded[:thr])

def pre_prompt(_author, data, subID_to_inst, ID_to_closest, situIndex_to_subID, num_shots, seed,
               fewshot_selftext, prompt_selftext, add_comment, add_Schwartz, add_ValueTradeoff, add_AbstValueTradeoff, add_ClusteredAbstValueTradeoff):

    judgment_to_word = {1: 'Acceptable', 0: 'Unacceptable'}

    all_prompts, all_IDs, all_judgments, all_seeds, all_evidence = [], [], [], [], []
    for i in range(len(data)):
        _ID = data[i]['subID'] + '###' + _author
        _closest_subIDs = [situIndex_to_subID[elem[0]] for elem in ID_to_closest[_ID]]
        _closest_subIDs = [elem for elem in _closest_subIDs if elem != data[i]['subID']]
        
        if seed != -1:
            np.random.seed(seed)
            all_author_subIDs = list(subID_to_inst.keys())
            np.random.shuffle(all_author_subIDs)
            _closest_subIDs = all_author_subIDs[:]

        instruction_msg = "You will be given examples of [Situation], ##_COMMENT_##, ##_SCHWARTZ_##, ##_VALUE_##, ##_ABST_##, ##_CLUSTERED_ABST_##, and [Person X's judgment on the situation (i.e. whether it is acceptable or not)].\n\n"

        if add_comment:
            instruction_msg = instruction_msg.replace('##_COMMENT_##', "[Person X's comment on the situation]")
        else:
            instruction_msg = instruction_msg.replace(', ##_COMMENT_##', "")
        if add_Schwartz:
            instruction_msg = instruction_msg.replace('##_SCHWARTZ_##', "[Person X's value on the situation]")
        else:
            instruction_msg = instruction_msg.replace(', ##_SCHWARTZ_##', "")
        if add_ValueTradeoff:
            instruction_msg = instruction_msg.replace('##_VALUE_##', "[Person X's value trade-off on the situation]")
        else:
            instruction_msg = instruction_msg.replace(', ##_VALUE_##', "")
        if add_AbstValueTradeoff:
            instruction_msg = instruction_msg.replace('##_ABST_##', "[Person X's value trade-off on the situation]")
        else:
            instruction_msg = instruction_msg.replace(', ##_ABST_##', "")
        if add_ClusteredAbstValueTradeoff:
            instruction_msg = instruction_msg.replace('##_CLUSTERED_ABST_##', "[Person X's value trade-off on the situation]")
        else:
            instruction_msg = instruction_msg.replace(', ##_CLUSTERED_ABST_##', "")

        _fewshot_situations = "###\n\n"

        for _subID in _closest_subIDs[:num_shots]:
            _inst = subID_to_inst[_subID]

            _fewshot_situations += f"[Situation] {_inst['title']}\n"
            if fewshot_selftext:
                _fewshot_situations += f"{truncate_text(_inst['selftext'], thr=512)}\n"

            if add_comment:
                _fewshot_situations += f"[Person X's comment] {_inst['comment']}\n"
            if add_Schwartz:
                _fewshot_situations += f"[Person X's value] {_inst['schwartz']}\n"
            if add_ValueTradeoff:
                _fewshot_situations += f"[Person X's value trade-off] {_inst['value']}\n"
            if add_AbstValueTradeoff:
                _fewshot_situations += f"[Person X's value trade-off] {_inst['abst_value']}\n"
            if add_ClusteredAbstValueTradeoff:
                _fewshot_situations += f"[Person X's value trade-off] {_inst['abst_value_clustered']}\n"

            _fewshot_situations += f"[Person X's judgment] {str(_inst['judgment'])} ({judgment_to_word[_inst['judgment']]})\n"
            _fewshot_situations += "\n###\n\n"

        task_msg = "Now you will be given a new situation. Based on your understanding of Person X's judgments on different situations, tell me how Person X would judge the new situation. ANSWER WITH NUMBERS ONLY (1 being Acceptable, 0 being Unacceptable), WITHOUT ANY OTHER EXPLANATIONS.\n"
        
        prompt_msg = f"[Situation] {data[i]['title']} {truncate_text(data[i]['selftext'])}\n[Person X's judgment] " if prompt_selftext else f"[Situation] {data[i]['title']}\n[Person X's judgment] "

        curr_prompt = instruction_msg + _fewshot_situations + task_msg + prompt_msg
        all_prompts.append(curr_prompt)
        all_IDs.append(data[i]['subID']+'###'+_author)
        all_judgments.append(data[i]['judgment'])
        all_evidence.append(_closest_subIDs[:num_shots])
        all_seeds.append(seed)

    assert len(all_prompts) == len(all_IDs) == len(all_judgments) == len(all_evidence)

    return all_prompts, all_IDs, all_judgments, all_seeds, all_evidence


def get_token_probability(outputs, target_token):
    """
    Extracts the probability of a specific target token from vLLM's output.
    """
    if not outputs:
        print(f"No outputs received for target token: {target_token}")
        return 0.0

    # Assuming we are interested in the first (and only) generated token
    first_output = outputs.outputs[0]
    if not first_output.logprobs:
        print("No logprobs found in the output.")
        return 0.0

    # Iterate through the log probabilities of the first token
    # The logprobs dictionary maps token ID to its log probability
    # We need to map token ID back to token string to find our target.
    # vLLM's output.logprobs is a list of dictionaries, one for each token position.
    # Since max_tokens=1, we only care about the first element of this list.
    first_token_logprobs = first_output.logprobs[0]

    # Get the tokenizer from the LLM instance to decode token IDs
    tokenizer = llm.get_tokenizer()

    for token_id, logprob_obj in first_token_logprobs.items(): # Renamed 'logprob' to 'logprob_obj' for clarity
        decoded_token = tokenizer.decode([token_id])
        # print(f"  Decoded token: '{decoded_token}', Logprob: {logprob_obj}") # For debugging

        # Check if the decoded token matches our target token.
        # Be mindful of potential leading spaces or capitalization from the tokenizer.
        if decoded_token.strip().lower() == target_token.strip().lower():
            # Access the actual float value from the logprob object
            return torch.exp(torch.tensor(logprob_obj.logprob)).item() # Changed here

    print(f"Target token '{target_token}' not found in top logprobs for this output.")
    return 0.0 # Return 0 if the target token is not in the top_logprobs

with open(f"{os.getcwd()}/data/author_to_subID_to_inst.pkl", "rb") as f:
    author_to_subID_to_inst = pickle.load(f)

with open(f"{os.getcwd()}/data/ID_to_closest_bySituation.pkl", "rb") as f:
    ID_to_closest = pickle.load(f)
with open(f'{os.getcwd()}/embedding/Situation_text-embedding-3-large.pkl', 'rb') as f:
    situ_dist_data = pickle.load(f)

subID_to_situIndex, situIndex_to_subID = {}, {}
for idx, elem in enumerate(situ_dist_data['ID']):
    subID_to_situIndex[elem] = idx
    situIndex_to_subID[idx] = elem

with open(f"{os.getcwd()}/data/author_to_controversialIndex_byRedditor.pkl", "rb") as f:
    author_to_testHardIdx = pickle.load(f)


author_cnt = {_author:len(author_to_testHardIdx[_author]) for _author in author_to_testHardIdx}
author_cnt = sorted(author_cnt.items(), key=lambda x:x[1], reverse=True)
AUTHORS = [elem[0] for elem in author_cnt[:100]]

# del situ_dist_data

num_shots = 10
all_prompts, all_IDs, all_judgments, all_seeds, all_evidence = [], [], [], [], []
for _author in AUTHORS[:8]:
    _train_set, _valid_set, _test_set = get_data_split_author('/data/byRedditor', _author, 10000, 10000, 10000, return_dict=False)
    
    for seed in [-1] + list(range(10)):
        _prompts, _IDs, _judg, _seed, _evidence = pre_prompt(_author, _train_set, author_to_subID_to_inst[_author], ID_to_closest, situIndex_to_subID, 
                                                         num_shots, seed, True, True, True, False, False, False, False)
        all_prompts += _prompts
        all_IDs += _IDs
        all_judgments += _judg
        all_seeds += _seed
        all_evidence += _evidence

sampling_params = SamplingParams(
    n=1,
    temperature=0.0, 
    top_p=1.0,
    max_tokens=5, # We only want one word: 'yes' or 'no'
    logprobs=10
)

model_path = "meta-llama/Llama-3.1-8B-Instruct"
llm = LLM(model=model_path, max_model_len=8192, dtype="auto")

all_outputs = llm.generate(all_prompts, sampling_params, use_tqdm=True)

all_probs1, all_probs0 = [], []
for _output in tqdm(all_outputs):
    all_probs1.append(get_token_probability(_output, "1"))
    all_probs0.append(get_token_probability(_output, "0"))

with open(f"{os.getcwd()}/vllm_inference/prob_per_different_evidence_withGold_llama-8b.pkl", "wb") as f:
    pickle.dump({"ID":all_IDs, "Seed":all_seeds, "Evidence":all_evidence, "prob1":all_probs1, "prob0":all_probs0}, f)