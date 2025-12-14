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

import wandb

from datasets import Dataset
from sklearn.metrics import *
from evaluate import load

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
from transformers import TrainingArguments, Trainer, DataCollatorWithPadding, EarlyStoppingCallback

from _00_data_loader import get_data_split_author
from _00_metrics import compute_macrof1, compute_accuracy


def hypSearch(
            model_ckpt,
            metric,
            tokenizer,
            max_ctx_len,
            n_labels,
            output_dir,
            input_type,
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
            add_instruction,
            train_batch_trials,
            learning_rate_trials):

    def preprocess_(batch):
        if input_type == 'selftext':
            INSTRUCTION = """Classify whether the given [SITUATION] is acceptable or not:\n##\n[SITUATION] """
            if add_instruction:
                inputs = [INSTRUCTION + title + "\n" + selftext for title, selftext in zip(batch["title"], batch["selftext"])]
            else:
                inputs = [title + "\n" + selftext for title, selftext in zip(batch["title"], batch["selftext"])]
        elif input_type == 'comment':
            INSTRUCTION = """Classify whether the [COMMENT] says a given [SITUATION] is acceptable or not:\n##\n[SITUATION] """
            if add_instruction:
                inputs = [INSTRUCTION + title + "\n[COMMENT] " + comment for title, comment in zip(batch["title"], batch["comment"])]
            else:
                inputs = [title + "\n" + comment for title, comment in zip(batch["title"], batch["comment"])]
        elif input_type == 'RoT':
            INSTRUCTION = """Classify whether the [PERSPECTIVE] says a given [SITUATION] is acceptable or not:\n##\n[SITUATION] """
            if add_instruction:
                inputs = [INSTRUCTION + title + '\n[PERSPECTIVE] ' + '\n'.join(RoT) for title, RoT in zip(batch["title"], batch["RoT"])]
            else:
                inputs = [title + '\n' + '\n'.join(RoT) for title, RoT in zip(batch["title"], batch["RoT"])]

        model_inputs = tokenizer(inputs, max_length=max_ctx_len, truncation=True)
        model_inputs["label"] = [judgment for judgment in batch["judgment"]]
        return model_inputs

    # training the model with Huggingface 🤗 trainer
    train_dataset = Dataset.from_dict(train_data)
    val_dataset = Dataset.from_dict(val_data)
    test_dataset = Dataset.from_dict(test_data)

    # tokenize the dataset
    train_dataset = train_dataset.map(preprocess_, batched=True,)
    val_dataset = val_dataset.map(preprocess_, batched=True,)
    test_dataset = test_dataset.map(preprocess_, batched=True,)

    if metric == "f1":
        compute_metrics = compute_macrof1
    elif metric == "accuracy":
        compute_metrics = compute_accuracy

    def model_init(trial):
        return AutoModelForSequenceClassification.from_pretrained(
            model_ckpt,
            from_tf=bool(".ckpt" in model_ckpt),
            token=None,
        )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=output_dir, 
        num_train_epochs=epochs,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=eval_batch_size,  
        eval_strategy='steps',
        learning_rate=learning_rate, 
        weight_decay=0.01,
        # fp16=True,
        logging_steps=len(train_dataset),
        eval_steps=len(train_dataset),
        save_steps=len(train_dataset), 
        logging_dir=os.path.join(output_dir, "runs/"),
        save_total_limit=save_total_limit,
        # save_strategy="no",
        seed=seed,
        load_best_model_at_end=True,
        report_to="wandb",
    )

    trainer = Trainer(
        model=None,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
        processing_class=tokenizer,
        model_init=model_init,
        data_collator=data_collator,
    )

    def wandb_hp_space(trial):
        return {
            "method": "random",
            "metric": {"name": "objective", "goal": "minimize"},
            "parameters": {
                "learning_rate": {"distribution": "uniform", "min": learning_rate_trials[0], "max": learning_rate_trials[-1]},
                "per_device_train_batch_size": {"values": train_batch_trials},
            },
        }
    print("Starting Trials")
    best_trials = trainer.hyperparameter_search( 
        direction=["minimize", "maximize"],
        backend="wandb",
        hp_space=wandb_hp_space,
        n_trials=20,
    )
    print(best_trials)

    return

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--n_train', type=int, default=10000)
    parser.add_argument('--n_val', type=int, default=5000)
    parser.add_argument('--n_test', type=int, default=5000)

    parser.add_argument('--max_ctx_len', type=int, default=512)
    parser.add_argument('--n_labels', type=int, default=2)

    parser.add_argument('--model', type=str, default="roberta-base",)
    # parser.add_argument('--output_dir', type=str, default="/u/scratch1/lee3401/Research/model_outputs/")
    parser.add_argument('--output_dir', type=str, default="/scratch/gilbreth/lee3401/hypSearch/")
    parser.add_argument('--load_strategy', type=str, default='load_best', choices=['load_best', 'load_last'])
    parser.add_argument('--evaluate_metric', type=str, default='f1', choices=['accuracy', 'f1'])
    parser.add_argument('--input_type', type=str, default="selftext")

    parser.add_argument('--train_batch_trials', nargs='+', type=int)
    parser.add_argument('--learning_rate_trials', nargs='+', type=float)

    parser.add_argument('--author_datadir', type=str, default='/data/byRedditor')

    parser.add_argument('--seed', type=int, default=2024)
    parser.add_argument('--k', type=int, default=5)
    parser.add_argument('--save_total_limit', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=8)
    parser.add_argument('--train_batch_size', type=int, default=16)
    parser.add_argument('--eval_batch_size', type=int, default=32)
    parser.add_argument('--save_steps', type=int, default=2000)
    parser.add_argument('--learning_rate', type=float, default=2e-5)
    parser.add_argument('--early_stopping_patience', type=int, default=6)

    parser.add_argument('--add_instruction', action='store_true')

    args, unknown = parser.parse_known_args()

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

    output_dir = f"{args.output_dir}/{args.model}_{args.input_type}"
    if args.add_instruction:
        output_dir += f"_instruction"
    
    f1 = load("f1")
    _author = "WebbieVanderquack"

    output_dir_author = f"{output_dir}_{args.evaluate_metric}/{_author}/"
    _train_set, _valid_set, _test_set = get_data_split_author(args.author_datadir, _author, args.n_train, args.n_val, args.n_test)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    hypSearch(
        model_ckpt=model_path,
        metric=args.evaluate_metric,
        tokenizer=tokenizer,
        max_ctx_len=args.max_ctx_len,
        n_labels=args.n_labels,
        output_dir=output_dir_author,
        input_type=args.input_type,
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
        add_instruction=args.add_instruction,
        train_batch_trials=args.train_batch_trials,
        learning_rate_trials=args.learning_rate_trials,
    )

if __name__ == "__main__":
    main()
