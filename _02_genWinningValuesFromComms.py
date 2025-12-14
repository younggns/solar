import os
import re
import pandas as pd
from tqdm import tqdm
import pickle

import openai
from openai import OpenAI
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)
from _00_text_utils import comment_process, situation_value_process
from _00_prompts import FIND_WINNING_VALUE_PROMPT

# Load your API key from an environment variable or secret management service
openai.api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI()

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def completion_with_backoff(**kwargs):
    return client.chat.completions.create(**kwargs)

df_ext = pd.read_csv(os.getcwd()+'/data/data_active_authorExt.tsv', sep='\t')
df_orig = pd.read_csv(os.getcwd()+'/data/data_active.tsv', sep='\t')
df_values = pd.read_csv(os.getcwd()+'/data/situation_dimensions.tsv', sep='\t')

subID_to_values = {}
for idx, row in df_values.iterrows():
    subID_to_values[row['subID']] = situation_value_process(row['dimensions'])

already_processed_ID = {}
for idx, row in df_orig.iterrows():
    already_processed_ID[row['subID']+'###'+row['commAuthor']]=True

datadict = {'subID':[], 'title':[], 'selftext':[], 'author':[], 'comment':[], 'judgment':[], 'situ_val':[], 'comm_val':[]}

try:
    print(f"Total instances to iterate: {len(df_ext)-len(list(already_processed_ID.keys()))}")
    for idx, row in tqdm(df_ext.iterrows()):
        _ID = row['subID']+'###'+row['commAuthor']
        if _ID in already_processed_ID:
            continue

        input_text = f"[Situation] {row['title']}\n{row['selftext']}\n"
        input_text += "[Conflicting Values]\n" + "\n".join(['- '+item for item in subID_to_values[row['subID']]])+"\n"
        input_text += f"[Comment] {comment_process(row['comment'])}\n"
        prompt = FIND_WINNING_VALUE_PROMPT.replace("__InputPlaceholder__",input_text)
        message_list = [{"role": "user", "content": prompt},]

        response = completion_with_backoff(model="gpt-4o", messages=message_list)

        datadict['subID'].append(row['subID'])
        datadict['title'].append(row['title'])
        datadict['selftext'].append(row['selftext'])
        datadict['author'].append(row['commAuthor'])
        datadict['comment'].append(comment_process(row['comment']))
        datadict['judgment'].append(row['judgment'])
        datadict['situ_val'].append(subID_to_values[row['subID']])
        datadict['comm_val'].append(response.choices[0].message.content)
        # datadict['schwartz'].append(response_s.choices[0].message.content)

except Exception as e:
    print(e)
    with open(os.getcwd()+f'/gpt_inference/directComms_withValuesRefined.pkl', 'wb') as fout:
        pickle.dump(datadict, fout)

df_ans = pd.DataFrame(data=datadict)
df_ans.to_csv(os.getcwd()+f'/gpt_inference/directComms_withValuesRefined.tsv', sep='\t', index=False)