import os
import shutil
from copy import deepcopy
import random
import pickle
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
from utils.data_utils import load_sequence_classification_lolo_dataset, SequenceClassificationDataset
from utils.metrics import sentiment_metrics_fn
from sklearn.metrics import classification_report

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

###
# modelling functions
###
def get_lr(args, optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def metrics_to_string(metric_dict):
    string_list = []
    for key, value in metric_dict.items():
        string_list.append('{}:{:.2f}'.format(key, value))
    return ' '.join(string_list)

if __name__ == "__main__":
    # Make sure cuda is deterministic
    torch.backends.cudnn.deterministic = True
    
    # Parse args
    args = get_parser()

    # create directory
    output_dir = '{}/{}/{}/{}/{}_{}_{}'.format(
        args["model_dir"], 
        args["dataset_name"],
        args["task"], 
        f'lolo_{args["lang"]}', 
        args['model_checkpoint'].replace('/','-'),
        args['seed'],
        args["num_sample"]
    )

    if not os.path.exists(output_dir + '/test_results.json'):
        os.makedirs(output_dir, exist_ok=True)
    elif args['force']:
        print(f'overwriting model directory `{output_dir}`')
    else:
        raise Exception(f'model result `{output_dir}/test_results.json` already exists, use --force if you want to overwrite the folder')

    # Set random seed
    set_seed(args['seed'])  # Added here for reproductibility    
        
    # Prepare derived args
    if args["task"] == 'senti':
        strlabel2int = {'negative': 0, 'neutral': 1, 'positive': 2}
    # elif args["task"] == 'lid':
    #     strlabel2int = {
    #         'indonesian': 0, 'balinese': 1, 'acehnese': 2, 'maduranese': 3, 'banjarese': 4, 'javanese': 5, 
    #         'buginese': 6, 'sundanese': 7, 'ngaju': 8, 'minangkabau': 9, 'toba_batak': 10, 'english': 11
    #     }
    else:
        raise ValueError(f'Unknown value `{args["task"]}` for key `--task`')
    
    args["num_labels"] = len(strlabel2int)

    # load model
    model, tokenizer, vocab_path, config_path = load_model(args)
    optimizer = optim.Adam(model.parameters(), lr=args['lr'])

    if args['device'] == "cuda":
        model = model.cuda()

    print("=========== TRAINING PHASE ===========")
    train_dataset, valid_dataset, test_dataset = load_sequence_classification_lolo_dataset(
        args["dataset_name"],
        args["task"],
        args["lang"],
        strlabel2int,
        tokenizer,
        args["text_column_name"],
        args["label_column_name"],
        args["num_sample"],
        args['seed']
    )
    print(f"len(train_dataset): {len(train_dataset)}")
    print(f"len(valid_dataset): {len(valid_dataset)}")
    print(f"len(test_dataset): {len(test_dataset)}")

    logging_dir = "logs"

    # Train
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
        evaluation_strategy='epoch',
        save_strategy="epoch",
        fp16=True,
        gradient_checkpointing=True,
        # logging_steps=logging_steps,
        eval_steps=1,
        save_steps=1,
        load_best_model_at_end = True,
        save_total_limit=1
    )

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
    valid_res = trainer.predict(valid_dataset)
    print(valid_res.metrics)

    ## -- Evaluation -- ##
    print("=========== EVALUATION PHASE ===========")
    eval_metrics = {}
    test_res = trainer.predict(test_dataset)
    eval_metrics[args["lang"]] = test_res.metrics
    print(f'Test results: {test_res.metrics}')

    # get prediction and true labels
    y_pred = test_res.predictions.argmax(axis=1)
    y_true = test_dataset.labels
    y_true = [strlabel2int[true_i] for true_i in y_true]

    # generate classification report
    cr = classification_report(y_true, y_pred, output_dict=True)
    cr_df = pd.DataFrame(cr).transpose()

    # saving final model
    trainer.save_model(f"{output_dir}/final_model")

    # save test results
    with open(f"{output_dir}/test_results.json", "w+") as f:
        json.dump({"valid": valid_res.metrics, "test": eval_metrics}, f)
        f.close()

    # save classification report
    cr_df.to_csv(f"{output_dir}/classification_report_df.csv")

    # save mapping of str labels to int 
    with open(f"{output_dir}/strlabel2int.json", "w+") as f:
        json.dump(strlabel2int, f)
        f.close()

    # save prediction results
    with open(f"{output_dir}/test_prediction_{args['lang']}.pkl", "wb") as f:
        pickle.dump(test_res, f)
        f.close()

    print("## -- Evaluation Done. -- ##")