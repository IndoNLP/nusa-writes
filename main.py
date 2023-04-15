import os
import shutil
from copy import deepcopy
import random
import json
import glob

import numpy as np
import pandas as pd
import torch
from torch import optim
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
from transformers import AdamW, AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from transformers import EarlyStoppingCallback
from nltk.tokenize import TweetTokenizer

from utils.functions import load_model, WordSplitTokenizer
from utils.args_helper import get_parser, print_opts
from utils.data_utils import load_sequence_classification_dataset, SequenceClassificationDataset, load_dataset
from utils.metrics import sentiment_metrics_fn

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def get_lr(args, optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def metrics_to_string(metric_dict):
    string_list = []
    for key, value in metric_dict.items():
        string_list.append('{}:{:.2f}'.format(key, value))
    return ' '.join(string_list)

if __name__ == "__main__":
    # Define Constants
    logging_dir = "logs"

    # Make sure cuda is deterministic
    torch.backends.cudnn.deterministic = True
    
    # Parse args
    args = get_parser()

    ## Helper 1: Create output directory
    def create_output_directory(model_dir, dataset_name, task, dataset_lang, model_checkpoint, seed, num_sample, force):
        output_dir = '{}/{}/{}/{}/{}_{}_{}'.format(
            model_dir,
            dataset_name,
            task,
            dataset_lang,
            model_checkpoint.replace('/','-'),
            seed,
            num_sample,
        )
        print(f"output_dir: {output_dir}")

        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        elif args['force']:
            print(f'overwriting model directory `{output_dir}`')
        else:
            raise Exception(f'model directory `{output_dir}` already exists, use --force if you want to overwrite the folder')
        return output_dir


    # Specify output dir
    output_dir = create_output_directory(
        args["model_dir"],
        args["dataset_name"],
        args["task"],
        args["lang"],
        args['model_checkpoint'].replace('/','-'),
        args['seed'],
        args["num_sample"],
        args["force"]
    )

    # Set random seed
    set_seed(args["seed"])

    # Load dset
    dset = load_dataset(
        dataset=args["dataset_name"],
        task=args["task"],
        lang=args["lang"],
        num_sample=int(args["num_sample"]),
        base_path="./data"
    )

    # Get unique labels
    unique_labels = list(set(dset["train"]["label"] + dset["valid"]["label"]))
    strlabel2int = {}
    for i, k in enumerate(unique_labels):
        strlabel2int[k] = i
    print(f"strlabel2int: {strlabel2int}")
    args["num_labels"] = len(strlabel2int)

    # load model
    model, tokenizer, vocab_path, config_path = load_model(args)
    optimizer = optim.Adam(model.parameters(), lr=args["lr"])

    # transfer model to GPU
    if args["device"] == "cuda":
        model = model.cuda()

    # Get train, valid and test split
    train_dataset, valid_dataset, test_dataset = load_sequence_classification_dataset(
        dset, strlabel2int, tokenizer, args["num_sample"], args["seed"]
    )
    print(f"len(train_dataset): {len(train_dataset)}")
    print(f"len(valid_dataset): {len(valid_dataset)}")
    print(f"len(test_dataset): {len(test_dataset)}")

    # Define Training Arguments
    training_args = TrainingArguments(
        output_dir=output_dir,          # output directory
        dataloader_num_workers=8,
        num_train_epochs=args["n_epochs"],              # total number of training epochs
        per_device_train_batch_size=args["train_batch_size"],  # batch size per device during training
        per_device_eval_batch_size=args["eval_batch_size"],   # batch size for evaluation
        learning_rate=args["lr"],              # number of warmup steps for learning rate scheduler
        weight_decay=args["gamma"],               # strength of weight decay
        gradient_accumulation_steps=args["grad_accum"], # Gradient accumulation
        logging_dir=logging_dir,            # directory for storing logs
        logging_strategy="epoch",
        evaluation_strategy='steps',
        save_strategy="steps",
        # logging_steps=logging_steps,
        eval_steps=150,
        save_steps=150,
        load_best_model_at_end = True,
        save_total_limit=1
    )

    # Train model
    trainer = Trainer(
        model=model, 
        args=training_args, 
        train_dataset=train_dataset, 
        eval_dataset=valid_dataset,
        tokenizer=tokenizer,
        compute_metrics=sentiment_metrics_fn,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )
    trainer.train()
    print("## -- Training Done. -- ##")
    valid_res = trainer.predict(valid_dataset)
    print(valid_res.metrics)

    # Evaluation
    print("=========== EVALUATION PHASE ===========")
    eval_metrics = {}
    test_res = trainer.predict(test_dataset)
    eval_metrics[args["lang"]] = test_res.metrics

    print(f'Test results: {test_res.metrics}')

    trainer.save_model(f"{output_dir}/final_model")
    log_output_path = output_dir + "/test_results.json"
    with open(log_output_path, "w+") as f:
        json.dump({"valid": valid_res.metrics, "test": eval_metrics}, f)

    print("## -- Evaluation Done. -- ##")





