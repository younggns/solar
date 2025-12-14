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
encoding = tiktoken.encoding_for_model("gpt-4o")

from vllm import LLM, SamplingParams


def truncate_text(text, thr=1024):
    if len(text) == 0:
        return text
    encoded = encoding.encode(text)
    return encoding.decode(encoded[:thr])

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

def get_fname(trial_idx, model_ckpt, num_shots, prompt_selftext, fewshot_selftext, do_CoT, rerank, add_comment, add_Schwartz, add_ValueTradeoff, add_AbstValueTradeoff, add_ClusteredAbstValueTradeoff, retrieval_strategy):
    if not os.path.exists(f"{os.getcwd()}/vllm_inference/{model_ckpt}"):
        os.makedirs(f"{os.getcwd()}/vllm_inference/{model_ckpt}")

    fname = f"{os.getcwd()}/vllm_inference/{model_ckpt}/inferJudgment_{num_shots}shots_{retrieval_strategy}"
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

def pre_prompt(_author, trial_idx, model_path, test_data, subID_to_inst, ID_to_closest, situIndex_to_subID, num_shots, prompt_selftext, fewshot_selftext, do_CoT, rerank, add_comment, add_Schwartz, add_ValueTradeoff, add_AbstValueTradeoff, add_ClusteredAbstValueTradeoff, retrieval_strategy):

    judgment_to_word = {1: 'Acceptable', 0: 'Unacceptable'}

    all_prompts, all_IDs, all_judgments = [], [], []
    for i in range(len(test_data)):

        _ID = test_data[i]['subID'] + '###' + _author
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

        curr_prompt = instruction_msg + _fewshot_situations + task_msg + prompt_msg
        all_prompts.append(curr_prompt)
        all_IDs.append(test_data[i]['subID']+'###'+_author)
        all_judgments.append(test_data[i]['judgment'])

    assert len(all_prompts) == len(all_IDs) == len(all_judgments)

    return all_prompts, all_IDs, all_judgments

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--n_train', type=int, default=10000)
    parser.add_argument('--n_val', type=int, default=5000)
    parser.add_argument('--n_test', type=int, default=5000)

    parser.add_argument('--n_trials', type=int, default=2)

    parser.add_argument('--model_ckpt', type=str, default='DeepSeek-V3') #
    parser.add_argument('--hardIndex_datapath', type=str, default='/data/author_to_controversialIndex_byRedditor.pkl')
    parser.add_argument('--author_datadir', type=str, default='/data/byRedditor')

    ## Retrieval Strategies
    parser.add_argument('--retrieval_strategy', type=str, default='Situation', choices=['Situation', 'Schwartz', 'Value', 'AbstValue'])

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
        
    subID_to_situIndex, situIndex_to_subID = {}, {}
    for idx, elem in enumerate(situ_dist_data['ID']):
        subID_to_situIndex[elem] = idx
        situIndex_to_subID[idx] = elem
    del situ_dist_data

    with open(f"{os.getcwd()}/{args.hardIndex_datapath}", "rb") as f:
        author_to_testHardIdx = pickle.load(f)

    if 'DeepSeek' in args.model_ckpt:
        model_path = f"deepseek-ai/{args.model_ckpt}"
    elif 'Qwen' in args.model_ckpt or 'QwQ' in args.model_ckpt:
        model_path = f"Qwen/{args.model_ckpt}"
    elif 'phi' in args.model_ckpt:
        model_path = f"microsoft/{args.model_ckpt}"
    elif 'Falcon' in args.model_ckpt:
        model_path = f"tiiuae/{args.model_ckpt}"
    elif 'Llama' in args.model_ckpt:
        model_path = f"meta-llama/{args.model_ckpt}"


    author_cnt = {_author:len(author_to_testHardIdx[_author]) for _author in author_to_testHardIdx}
    author_cnt = sorted(author_cnt.items(), key=lambda x:x[1], reverse=True)
    AUTHORS = [elem[0] for elem in author_cnt[:100]]

    all_labels, all_preds = [], []

    for i in range(args.n_trials):
        fname = get_fname(i, args.model_ckpt, args.num_shots, args.prompt_selftext, args.fewshot_selftext, args.do_CoT, args.rerank, args.add_comment, args.add_Schwartz, args.add_ValueTradeoff, args.add_AbstValueTradeoff, args.add_ClusteredAbstValueTradeoff, args.retrieval_strategy)

        if os.path.exists(f'{fname}.tsv'):
            df_ans = pd.read_csv(f'{fname}.tsv', sep='\t')
            print(classification_report(df_ans['gold'].tolist(), df_ans['pred'].tolist(), target_names=['Not Acceptable','Acceptable'], digits=4))
            continue

        all_prompts, all_IDs, all_judgments = [], [], []

        for _author in AUTHORS:
            _train_set, _valid_set, _test_set = get_data_split_author(args.author_datadir, _author, args.n_train, args.n_val, args.n_test, return_dict=False)
            
            author_prompts, author_IDs, author_judgments = pre_prompt(_author, i, model_path, _test_set, author_to_subID_to_inst[_author], ID_to_closest, situIndex_to_subID, args.num_shots, args.prompt_selftext, args.fewshot_selftext, args.do_CoT, args.rerank, args.add_comment, args.add_Schwartz, args.add_ValueTradeoff, args.add_AbstValueTradeoff, args.add_ClusteredAbstValueTradeoff, args.retrieval_strategy)

            all_prompts += author_prompts
            all_IDs += author_IDs
            all_judgments += author_judgments
        
        print("Done Loading")
        sampling_params = SamplingParams(temperature=0.1, min_p=0.05)
        # sampling_params = SamplingParams(temperature=0.6, top_p=0.95, min_p=0)
        # llm = LLM(model=model_path, max_model_len=800)
        llm = LLM(model=model_path, max_model_len=8192)
        all_outputs = llm.generate(all_prompts, sampling_params, use_tqdm=True)

        # with open(f'{fname}.pkl', 'wb') as f:
        #     pickle.dump(all_outputs, f)

        all_output_text = [elem.outputs[0].text for elem in all_outputs]
        all_output_digits = convert_pred_to_digit(all_judgments, all_output_text)

        assert len(all_output_digits)==len(all_judgments)==len(all_IDs)

        # print(f"\n[[Fold_{i}]]")
        # print(classification_report(all_judgments, all_output_digits, target_names=['Not Acceptable','Acceptable'], digits=4))

        df_ans = pd.DataFrame(data={'ID':all_IDs, 'pred':all_output_digits, 'gold':all_judgments})
        df_ans.to_csv(f'{fname}.tsv', sep='\t', index=False)

        all_labels += all_judgments
        all_preds += all_output_digits

    print(f"\n[[OVERALL]]")
    all_preds_refined = []
    for _l, _p in zip(all_labels, all_preds):
        if _p not in [0,1]:
            all_preds_refined.append(1-_l)
        else:
            all_preds_refined.append(_p)
    print(classification_report(all_labels, all_preds_refined, target_names=['Not Acceptable','Acceptable'], digits=4))


if __name__ == "__main__":
    main()
