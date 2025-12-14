import os
import re
import pandas as pd
from tqdm import tqdm
import pickle
import argparse
from _00_data_loader import get_data_split_author

from sklearn.metrics import *

import openai
from openai import OpenAI
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)

import tiktoken
encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")

def truncate_text(text, thr=1024):
    if len(text) == 0:
        return text
    encoded = encoding.encode(text)
    return encoding.decode(encoded[:thr])

# Load your API key from an environment variable or secret management service
openai.api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI()

deepseek_client = OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com")

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def completion_with_backoff(**kwargs):
    return client.chat.completions.create(**kwargs)

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def completion_with_backoff_deepseek(**kwargs):
    return deepseek_client.chat.completions.create(**kwargs)

def convert_pred_to_digit(labels, preds):
    pred_digits = []
    for _label, _pred in zip(labels, preds):
        if _pred.strip().isdigit():
            pred_digits.append(int(_pred.strip()))
        else:
            if '1' in _pred and '0' not in _pred:
                pred_digits.append(1)
            elif '0' in _pred and '1' not in _pred:
                pred_digits.append(0)
            elif 'acceptable' in _pred.lower() and 'unacceptable' not in _pred.lower():
                pred_digits.append(1)
            elif 'unacceptable' in _pred.lower() and 'Acceptable' not in _pred:
                pred_digits.append(0)
            else:
                pred_digits.append(1-_label)
            
    return pred_digits

def get_fname(_author, trial_idx, model_ckpt, num_shots, prompt_selftext, fewshot_selftext, do_CoT, rerank, add_comment, add_Schwartz, add_ValueTradeoff, add_AbstValueTradeoff, add_ClusteredAbstValueTradeoff, retrieval_strategy):
    if not os.path.exists(f"{os.getcwd()}/gpt_inference/{model_ckpt}"):
        os.makedirs(f"{os.getcwd()}/gpt_inference/{model_ckpt}")

    fname = f"{os.getcwd()}/gpt_inference/{model_ckpt}/{_author}_inferJudgment_{num_shots}shots_{retrieval_strategy}"
    if prompt_selftext:
        fname += "withSelftext_"
    if fewshot_selftext:
        fname += "withFewshotSelftext_"
    if add_comment:
        fname += "Comment_"
    if add_Schwartz:
        fname += "Schwartz_"
    if add_ValueTradeoff:
        fname += "Value_"
    if add_AbstValueTradeoff:
        fname += "AbstValue_"
    if add_ClusteredAbstValueTradeoff:
        fname += "AbstValueClustered_"
    if do_CoT:
        fname += "CoT_"
    if rerank != 'none':
        fname += f"{rerank}_"

    fname += f"trial{trial_idx}"

    return fname

def prompt(_author, trial_idx, model_ckpt, test_data, subID_to_inst, ID_to_closest, situIndex_to_subID, num_shots, prompt_selftext, fewshot_selftext, do_CoT, rerank, add_comment, add_Schwartz, add_ValueTradeoff, add_AbstValueTradeoff, add_ClusteredAbstValueTradeoff, retrieval_strategy):

    fname = get_fname(_author, trial_idx, model_ckpt, num_shots, prompt_selftext, fewshot_selftext, do_CoT, rerank, add_comment, add_Schwartz, add_ValueTradeoff, add_AbstValueTradeoff, add_ClusteredAbstValueTradeoff, retrieval_strategy)
    # print(fname)
    if os.path.exists(f'{fname}.tsv'):
        df_ans = pd.read_csv(f'{fname}.tsv', sep='\t')
        print(f"\n[[{_author}]]")
        print(classification_report(df_ans['gold'].tolist(), df_ans['pred'].tolist(), target_names=['Not Acceptable','Acceptable'], digits=4))

        return df_ans['gold'].tolist(), df_ans['pred'].tolist()

    results = {'subID':[], 'gold':[], 'pred':[]}
    judgment_to_word = {1: 'Acceptable', 0: 'Unacceptable'}

    for i in tqdm(range(len(test_data))):

        _ID = test_data[i]['subID'] + '###' + _author
        if len(retrieval_strategy.split('-')) > 1:
            _closest_IDs = [situIndex_to_subID[elem[0]] for elem in ID_to_closest[_ID]]
            _closest_subIDs = [elem.split('###')[0] for elem in _closest_IDs]
        else:
            _closest_subIDs = [situIndex_to_subID[elem[0]] for elem in ID_to_closest[_ID]]
        _closest_subIDs = [elem for elem in _closest_subIDs if elem != test_data[i]['subID']]

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
                _fewshot_situations += f"{_inst['selftext']}\n"

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

        task_msg = "Now you will be given a new situation. Based on your understanding of Person X's judgments on different situations, tell me how Person X would judge the new situation. ANSWER WITH NUMBERS ONLY (1 being Acceptable, 0 being Unacceptable), WITHOUT ANY OTHER EXPLANATIONS."
        
        prompt_msg = f"[Situation] {test_data[i]['title']} {truncate_text(test_data[i]['selftext'])}\n" if prompt_selftext else f"[Situation] {test_data[i]['title']}\n"

        message_list = [
            {"role": "user", "content": instruction_msg},
            {"role": "user", "content": _fewshot_situations},
            {"role": "user", "content": task_msg},
            {"role": "user", "content": prompt_msg},
        ]

        # for elem in message_list:
        #     print(elem['content'])
        # if i > 5:
        #     break

        if 'gpt' in model_ckpt:
            response = completion_with_backoff(model=model_ckpt, messages=message_list)
        else:
            response = completion_with_backoff_deepseek(model=model_ckpt, messages=message_list, stream=False)

        results['subID'].append(test_data[i]['subID'])
        results['gold'].append(test_data[i]['judgment'])
        results['pred'].append(response.choices[0].message.content)

    pred_digits = convert_pred_to_digit(results['gold'], results['pred'])

    try:
        df_ans = pd.DataFrame(data=results)
        fname = get_fname(_author, trial_idx, model_ckpt, num_shots, prompt_selftext, fewshot_selftext, do_CoT, rerank, add_comment, add_Schwartz, add_ValueTradeoff, add_AbstValueTradeoff, add_ClusteredAbstValueTradeoff, retrieval_strategy)
        df_ans.to_csv(f'{fname}.tsv', sep='\t', index=False)
    except:
        with open(f'{fname}.pkl', 'wb') as f:
            pickle.dump(results, f)

    print(f"\n[[{_author}]]")
    print(classification_report(results['gold'], pred_digits, target_names=['Not Acceptable','Acceptable'], digits=4))

    return results['gold'], pred_digits
    

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--n_train', type=int, default=10000)
    parser.add_argument('--n_val', type=int, default=5000)
    parser.add_argument('--n_test', type=int, default=5000)

    parser.add_argument('--n_trials', type=int, default=2)

    parser.add_argument('--model_ckpt', type=str, default='gpt-4.1', choices=['gpt-4.1-mini', 'gpt-4.1', 'gpt-4.1-nano', 'gpt-4o', 'gpt-4o-mini', 'deepseek-chat']) #
    parser.add_argument('--hardIndex_datapath', type=str, default='/data/author_to_controversialIndex_byRedditor.pkl')
    parser.add_argument('--author_datadir', type=str, default='/data/byRedditor')

    ## Retrieval Strategies
    parser.add_argument('--retrieval_strategy', type=str, default='Situation')

    ## Prompting Strategies
    parser.add_argument('--num_shots', type=int, default=5)
    parser.add_argument('--prompt_selftext', action='store_true')
    parser.add_argument('--fewshot_selftext', action='store_true')
    parser.add_argument('--do_CoT', action='store_true')
    parser.add_argument('--rerank', type=str, default='none', choices=['none', 'reverse', 'firstlast'])

    ## Indexed Documents
    parser.add_argument('--add_comment', action='store_true')
    parser.add_argument('--add_Schwartz', action='store_true')
    parser.add_argument('--add_ValueTradeoff', action='store_true')
    parser.add_argument('--add_AbstValueTradeoff', action='store_true')
    parser.add_argument('--add_ClusteredAbstValueTradeoff', action='store_true')

    args, unknown = parser.parse_known_args()

    with open(f"{os.getcwd()}/data/author_to_subID_to_inst.pkl", "rb") as f:
        author_to_subID_to_inst = pickle.load(f)
    
    with open(f"{os.getcwd()}/data/ID_to_closest_by{args.retrieval_strategy}.pkl", "rb") as f:
        ID_to_closest = pickle.load(f)
    with open(f'{os.getcwd()}/embedding/{args.retrieval_strategy}_text-embedding-3-large.pkl', 'rb') as f:
        situ_dist_data = pickle.load(f)

    situIndex_to_subID = {}
    for idx, elem in enumerate(situ_dist_data['ID']):
        situIndex_to_subID[idx] = elem
    del situ_dist_data

    with open(f"{os.getcwd()}/{args.hardIndex_datapath}", "rb") as f:
        author_to_testHardIdx = pickle.load(f)

    author_cnt = {_author:len(author_to_testHardIdx[_author]) for _author in author_to_testHardIdx}
    author_cnt = sorted(author_cnt.items(), key=lambda x:x[1], reverse=True)
    AUTHORS = [elem[0] for elem in author_cnt[:100]]

    all_labels, all_preds = [], []

    for i in range(args.n_trials):
        fold_labels, fold_preds = [], []
        for _author in AUTHORS:
            _train_set, _valid_set, _test_set = get_data_split_author(args.author_datadir, _author, args.n_train, args.n_val, args.n_test, return_dict=False)
            
            author_labels, author_preds = prompt(_author, i, args.model_ckpt, _test_set, author_to_subID_to_inst[_author], ID_to_closest, situIndex_to_subID, args.num_shots, args.prompt_selftext, args.fewshot_selftext, args.do_CoT, args.rerank, args.add_comment, args.add_Schwartz, args.add_ValueTradeoff, args.add_AbstValueTradeoff, args.add_ClusteredAbstValueTradeoff, args.retrieval_strategy)

            fold_labels += author_labels
            fold_preds += author_preds

        print(f"\n[[Fold_{i}]]")
        print(classification_report(fold_labels, fold_preds, target_names=['Not Acceptable','Acceptable'], digits=4))

        all_labels += fold_labels
        all_preds += fold_preds

    print(f"\n[[OVERALL]]")
    print(classification_report(all_labels, all_preds, target_names=['Not Acceptable','Acceptable'], digits=4))


if __name__ == "__main__":
    main()
