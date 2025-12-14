import os
import gc
import argparse
import pickle
from collections import Counter
from tqdm import tqdm
import shutil
import time

import pandas as pd
import numpy as np

import nltk
import wandb

from datasets import Dataset
from evaluate import load
from sklearn.metrics import *

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoConfig
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq
from peft import LoraConfig, TaskType, get_peft_model

# from _00_metrics import compute_rouge, compute_bertscore
from _00_data_loader import get_data_split_author
from _00_metrics import compute_macrof1, compute_accuracy
from _00_custom_trainer import WeightedTrainer

def load_model(output_dir, model_path, strategy):
    if output_dir and os.path.exists(output_dir):
        if "checkpoint" in output_dir:
            model = AutoModelForSeq2SeqLM.from_pretrained(output_dir)
            tokenizer = AutoTokenizer.from_pretrained(output_dir)
        else:
            if strategy == 'load_last':
                print("Loading last model from checkpoint")
                latest_checkpoint_idx = 0
                dir_list = os.listdir(output_dir) # find the latest checkpoint
                for d in dir_list:
                    if "checkpoint" in d and "best" not in d:
                        checkpoint_idx = int(d.split("-")[-1])
                        if checkpoint_idx > latest_checkpoint_idx:
                            latest_checkpoint_idx = checkpoint_idx
                if latest_checkpoint_idx > 0 and os.path.exists(os.path.join(output_dir, f"checkpoint-{latest_checkpoint_idx}")):
                    ft_model_path = os.path.join(output_dir, f"checkpoint-{latest_checkpoint_idx}")
                    model = AutoModelForSeq2SeqLM.from_pretrained(ft_model_path)
                    tokenizer = AutoTokenizer.from_pretrained(ft_model_path)
                    return model, tokenizer
            elif strategy == 'load_best':
                print("Loading best model from checkpoint")
                ft_model_path = os.path.join(output_dir, f"best_checkpoint")
                if os.path.exists(ft_model_path):
                    model = AutoModelForSeq2SeqLM.from_pretrained(ft_model_path)
                    tokenizer = AutoTokenizer.from_pretrained(ft_model_path)
                    return model, tokenizer
                else:
                    non_latest_checkpoint_idx = 100000
                    dir_list = os.listdir(output_dir) # find the non-latest checkpoint
                    for d in dir_list:
                        if "checkpoint" in d:
                            checkpoint_idx = int(d.split("-")[-1])
                            if checkpoint_idx < non_latest_checkpoint_idx:
                                non_latest_checkpoint_idx = checkpoint_idx
                    if non_latest_checkpoint_idx > 0 and os.path.exists(os.path.join(output_dir, f"checkpoint-{non_latest_checkpoint_idx}")):
                        ft_model_path = os.path.join(output_dir, f"checkpoint-{non_latest_checkpoint_idx}")
                        model = AutoModelForSeq2SeqLM.from_pretrained(ft_model_path)
                        tokenizer = AutoTokenizer.from_pretrained(ft_model_path)
                        return model, tokenizer
            else:
                raise NotImplementedError

    # load pretrained model for hf
    print(f"Loading HF pretrained model {model_path}")
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return model, tokenizer

def fine_tune_hf(
            model,
            model_ckpt,
            metric,
            tokenizer,
            max_ctx_len,
            max_tgt_len,
            output_dir,
            output_type,
            train_data,
            val_data,
            test_data,
            epochs,
            train_batch_size,
            eval_batch_size,
            save_total_limit,
            early_stopping_patience,
            learning_rate,
            seed,
            do_label_weighting,
            do_lora,
            lora_r,
            lora_a,
            lora_target_modules,
            add_instruction,
            do_train,
            do_inference):

    def preprocess_(batch):
        _mapping = {1:"NTA", 0:"YTA"}
        if add_instruction:
            if output_type == 'comment':
                INSTRUCTION = """Generate a judgment (YTA or NTA) and comment about the input [SITUATION]:\n##\n[SITUATION] """
            elif output_type == 'RoT':
                INSTRUCTION = """Generate a likely perspective about the input [SITUATION]:\n##\n[SITUATION] """
            inputs = [INSTRUCTION + title + "\n" + selftext for title, selftext in zip(batch["title"], batch["selftext"])]
        else:
            inputs = [title + "\n" + selftext for title, selftext in zip(batch["title"], batch["selftext"])]
        model_inputs = tokenizer(inputs, max_length=max_ctx_len, padding=True, truncation=True)

        if output_type == 'comment':
            targets = [f"[Judgment] {_mapping[judgment]}\n[Comment] {comment}" for judgment, comment in zip(batch["judgment"], batch["comment"])]
        elif output_type == 'RoT':
            targets = ['\n'.join(RoT) for RoT in batch["RoT"]]
        labels = tokenizer(targets, max_length=max_tgt_len, truncation=True, padding=True, )
        model_inputs["labels"] = labels["input_ids"]

        return model_inputs

    # training the model with Huggingface 🤗 trainer
    train_dataset = Dataset.from_dict(train_data)
    val_dataset = Dataset.from_dict(val_data)
    test_dataset = Dataset.from_dict(test_data)

    # tokenize the dataset
    train_dataset = train_dataset.map(preprocess_, batched=True,)
    val_dataset = val_dataset.map(preprocess_, batched=True,)
    test_dataset = test_dataset.map(preprocess_, batched=True,)
    # print(test_dataset)

    if metric == "rouge":
        rouge = load("rouge")
    elif metric == "bertscore":
        bertscore = load("bertscore")
    

    def compute_rouge(eval_pred):
        predictions, labels = eval_pred
        # Replace -100 in the labels as we can't decode them.
        preds= np.where(predictions != -100, predictions, tokenizer.pad_token_id)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        # Rouge expects a newline after each sentence
        try:
            decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
            decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]
        except:
            decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
            decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]
        
        result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
        # Extract a few results
        # result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
        
        # Add mean generated length
        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
        result["gen_len"] = np.mean(prediction_lens)
        
        return result

    def compute_bertscore(eval_pred):
        predictions, labels = eval_pred
        # Replace -100 in the labels as we can't decode them.
        preds= np.where(predictions != -100, predictions, tokenizer.pad_token_id)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)    
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # result = bertscore.compute(predictions=decoded_preds, references=decoded_labels, lang="en", idf=True, model_type="distilbert-base-uncased")
        result = bertscore.compute(predictions=decoded_preds, references=decoded_labels, lang="en", idf=True)
        
        # Add mean generated length
        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
        result["gen_len"] = np.mean(prediction_lens)
        
        return result

    if metric == "rouge":
        compute_metrics = compute_rouge
    elif metric == "bertscore":
        compute_metrics = compute_bertscore

    logging_steps = int(len(train_dataset)/train_batch_size)

    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir, 
        num_train_epochs=epochs,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=eval_batch_size,  
        eval_strategy='steps',
        learning_rate=learning_rate, 
        weight_decay=0.01,
        # fp16=True,
        logging_steps=logging_steps,
        eval_steps=logging_steps,
        save_steps=logging_steps, 
        logging_dir=os.path.join(output_dir, "runs/"),
        save_total_limit=save_total_limit,
        seed=seed,
        # save_strategy="no",
        load_best_model_at_end=True,
        report_to="none",
        predict_with_generate=True, # for seq2seq trainer
        generation_max_length=max_tgt_len, # for seq2seq trainer
    )
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer)
    if do_lora:
        lora_config = LoraConfig(task_type=TaskType.CAUSAL_LM, r=lora_r, lora_alpha=lora_a, lora_dropout=0.1, 
                                target_modules=lora_target_modules,)
        model = get_peft_model(model, lora_config)

    trainer = Seq2SeqTrainer(
        model=model, 
        tokenizer=tokenizer, 
        args=training_args, 
        train_dataset=train_dataset, 
        eval_dataset=val_dataset, 
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    if do_train:
        train_results = trainer.train()

    if do_inference:
        test_results = trainer.predict(test_dataset)
        # predictions, labels = test_results[0], test_results[1]

        # preds= np.where(predictions != -100, predictions, tokenizer.pad_token_id)
        # labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        # decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        # decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # if metric == "rouge":
        #     try:
        #         decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
        #         decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]
        #     except:
        #         nltk.download('punkt')
        #         decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
        #         decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]
        # print(decoded_preds)
        # print(decoded_labels)

        # results = compute_metrics(test_results)

    model.cpu()
    del model
    del tokenizer
    del trainer
    del train_dataset
    del val_dataset
    del test_dataset
    gc.collect()
    torch.cuda.empty_cache()

    return test_results if do_inference else train_results

def main():
    nltk.download('punkt')
    parser = argparse.ArgumentParser()

    parser.add_argument('--n_train', type=int, default=10000)
    parser.add_argument('--n_val', type=int, default=5000)
    parser.add_argument('--n_test', type=int, default=5000)

    parser.add_argument('--max_ctx_len', type=int, default=512)
    parser.add_argument('--max_tgt_len', type=int, default=512)

    parser.add_argument('--model', type=str, default="roberta-base",)
    parser.add_argument('--output_dir', type=str, default="/scratch/gilbreth/lee3401/sft_baseline_seq2seq/")
    parser.add_argument('--load_strategy', type=str, default='load_best', choices=['load_best', 'load_last'])
    parser.add_argument('--evaluate_metric', type=str, default='rouge', choices=['rouge', 'bertscore'])
    parser.add_argument('--output_type', type=str, default="comment")

    parser.add_argument('--hardIndex_datapath', type=str, default='/data/author_to_controversialIndex_byRedditor.pkl')
    parser.add_argument('--author_datadir', type=str, default='/data/byRedditor')
    parser.add_argument('--num_hardInst_thr', type=int, default=20)

    parser.add_argument('--seed', type=int, default=2024)
    parser.add_argument('--k', type=int, default=5)
    parser.add_argument('--save_total_limit', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--train_batch_size', type=int, default=32)
    parser.add_argument('--eval_batch_size', type=int, default=32)
    parser.add_argument('--save_steps', type=int, default=2000)
    parser.add_argument('--learning_rate', type=float, default=3.97838e-5)
    parser.add_argument('--early_stopping_patience', type=int, default=7)

    parser.add_argument('--do_label_weighting', action='store_true')
    parser.add_argument('--do_lora', action='store_true')
    parser.add_argument('--lora_r', type=int, default=4)
    parser.add_argument('--lora_a', type=int, default=16)

    parser.add_argument('--add_instruction', action='store_true')

    parser.add_argument('--do_train', action='store_true')
    parser.add_argument('--do_inference', action='store_true')
    args, unknown = parser.parse_known_args()

    lora_target_modules = []
    if 'deberta' in args.model or 'Phi-3' in args.model:
        lora_target_modules = ["query_proj", "value_proj"]
        model_path = f"microsoft/{args.model}"
    elif 'bart' in args.model:
        model_path = f"facebook/{args.model}"
        lora_target_modules = ["q_proj", "v_proj"]
    elif 'flan-t5' in args.model or 'gemma' in args.model:
        model_path = f"google/{args.model}"
        lora_target_modules = ["q", "v"]
    elif 'Qwen' in args.model:
        model_path = f"Qwen/{args.model}"
        lora_target_modules = ["q_proj", "v_proj"]
    elif 'distilbert' in args.model:
        model_path = f"distilbert/{args.model}"
        lora_target_modules = ["q_lin", "v_lin"]
    else:
        model_path = args.model
        lora_target_modules = ["query", "value"]

    output_dir = f"{args.output_dir}/{args.model}_{args.output_type}"
    if args.do_label_weighting:
        output_dir += f"_weightedlabels"
    if args.do_lora:
        output_dir += f"_LoRA_r{args.lora_r}a{args.lora_a}"
    if args.add_instruction:
        output_dir += f"_instruction"

    with open(f"{os.getcwd()}/{args.hardIndex_datapath}", "rb") as f:
        author_to_testHardIdx = pickle.load(f)

    author_cnt = {_author:len(author_to_testHardIdx[_author]) for _author in author_to_testHardIdx}
    author_cnt = sorted(author_cnt.items(), key=lambda x:x[1], reverse=True)
    AUTHORS = [elem[0] for elem in author_cnt[:100]]

    all_labels, all_preds = [], []
    all_labels_hard, all_preds_hard = [], []
    author_results, author_hard_results = {_author:[] for _author in AUTHORS}, {_author:[] for _author in AUTHORS}

    for i in range(args.k):
        fold_labels, fold_preds = [], []
        print("================================================")
        print(f"=====================fold{i}======================")
        print("================================================")
        for _author in AUTHORS:
            print(f"[AUTHOR] {_author}")
            output_dir_author = f"{output_dir}_{args.evaluate_metric}_fold{i}/{_author}/"
            _train_set, _valid_set, _test_set = get_data_split_author(args.author_datadir, _author, args.n_train, args.n_val, args.n_test, fold=i)
            model, tokenizer = load_model(output_dir_author, model_path, args.load_strategy)

            print(f"\n[[AUTHOR]] {_author}")

            results = fine_tune_hf(
                model=model,
                model_ckpt=args.model,
                metric=args.evaluate_metric,
                tokenizer=tokenizer,
                max_ctx_len=args.max_ctx_len,
                max_tgt_len=args.max_tgt_len,
                output_dir=output_dir_author,
                output_type=args.output_type,
                train_data=_train_set,
                val_data=_valid_set,
                test_data=_test_set,
                epochs=args.epochs,
                train_batch_size=args.train_batch_size,
                eval_batch_size=args.eval_batch_size,
                save_total_limit=args.save_total_limit,
                early_stopping_patience=args.early_stopping_patience,
                learning_rate=args.learning_rate,
                seed=args.seed,
                do_label_weighting=args.do_label_weighting,
                do_lora=args.do_lora,
                lora_r=args.lora_r,
                lora_a=args.lora_a,
                lora_target_modules=lora_target_modules,
                add_instruction=args.add_instruction,
                do_train=args.do_train,
                do_inference=args.do_inference
            )

            predictions, labels = results[0], results[1]
            preds= np.where(predictions != -100, predictions, tokenizer.pad_token_id)
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
            decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
            decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)


            if args.evaluate_metric == "rouge":
                try:
                    decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
                    decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]
                except:
                    nltk.download('punkt')
                    decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
                    decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]
            
            pred_digits, label_digits = [], []
            for _pred, _label in zip(decoded_preds, decoded_labels):
                if '[Judgment] NTA' in _label:
                    label_digits.append(1)
                elif '[Judgment] YTA' in _label:
                    label_digits.append(0)
                else:
                    raise NotImplementedError
                if 'nta' in [item.lower() for item in _pred.split()]:
                    pred_digits.append(1)
                elif 'yta' in [item.lower() for item in _pred.split()]:
                    pred_digits.append(0)
                else:
                    _curr_pred = label_digits[-1]
                    pred_digits.append(1-_curr_pred)

            fold_labels += label_digits
            fold_preds += pred_digits

            _fold_author_results = classification_report(label_digits, pred_digits, target_names=['Not Acceptable','Acceptable'], output_dict=True)
            author_results[_author].append(_fold_author_results)

            # num_hard_cases = author_to_testHardIdx[_author].count(1)
            # if num_hard_cases >= args.num_hardInst_thr:
            _hard_labels, _hard_preds = [], []
            for _l, _p, _isHard in zip(label_digits, pred_digits, author_to_testHardIdx[_author]):
                if _isHard == 1:
                    _hard_labels.append(_l)
                    _hard_preds.append(_p)
            # _fold_author_hard_results = classification_report(_hard_labels, _hard_preds, target_names=['Not Acceptable','Acceptable'], output_dict=True)
            author_hard_results[_author].append([_hard_labels, _hard_preds])

            all_labels_hard += _hard_labels
            all_preds_hard += _hard_preds

            model.cpu()
            del model
            del tokenizer
            del _train_set
            del _valid_set
            del _test_set
            gc.collect()
            torch.cuda.empty_cache()

        print(f"\n[[Fold_{i} OVERALL]]")
        print(classification_report(fold_labels, fold_preds, target_names=['Not Acceptable','Acceptable'], digits=4))

        all_labels += fold_labels
        all_preds += fold_preds

    with open(f"{os.getcwd()}/outputs/{args.model}_seq2seq_author_results.pkl", "wb") as f:
        pickle.dump(author_results, f)
    with open(f"{os.getcwd()}/outputs/{args.model}_seq2seq_author_results_hard.pkl", "wb") as f:
        pickle.dump(author_hard_results, f)
    
    print(f"\n[[OVERALL]]")
    print(classification_report(all_labels, all_preds, target_names=['Not Acceptable','Acceptable'], digits=4))

    print(f"\n[[OVERALL-Hard]]")
    print(classification_report(all_labels_hard, all_preds_hard, target_names=['Not Acceptable','Acceptable'], digits=4))
        

if __name__ == "__main__":
    main()
