import os
import shutil
import pickle
import torch
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch import optim
from copy import deepcopy
from utils.data_utils import load_dataset
from torch.optim.lr_scheduler import StepLR
from utils.args_helper import (
    get_generation_parser, 
    append_generation_dataset_args, 
    append_generation_model_args
)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

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

def evaluate_classical(list_hyp, list_label, metrics_fn):
    metrics = metrics_fn(list_hyp, list_label)        
    return None, metrics, list_hyp, list_label

def translate_one_sentence_panlex(
    sentence,
    translator
):
    words = sentence.split(' ')
    translated_words = []
    for word in words:
        tranlated_word = translator.get(word)
        if tranlated_word:
            translated_words.append(tranlated_word)
        else:
            translated_words.append(word)
    translated_sentence = ' '.join(translated_words)
    return translated_sentence

def translate_lexical_panlex(
    sentences,
    src_lang,
    dst_lang
):
    translator_filepath = f"./panlex_translator/{src_lang}_to_{dst_lang}.pkl"
    with open(translator_filepath, 'rb') as fp:
        translator = pickle.load(fp)
    return [translate_one_sentence_panlex(s, translator) for s in sentences]

def process_classical_benchmark(args):
    # Specify output dir
    output_dir = create_output_directory(
        args["model_dir"],
        args["dataset_name"],
        args["task"],
        args["lang"],
        args['model_type'].replace('/','-'),
        args['seed'],
        args["num_sample"],
        args["force"]
    )

    # Set random seed
    set_seed(args['seed'])  # Added here for reproductibility    
    
    metrics_scores = []
    result_dfs = []

    # Load dset
    dset = load_dataset(
        dataset=args["dataset_name"],
        task=args["task"],
        lang=args["lang"],
        num_sample=int(args["num_sample"]),
        base_path="./data"
    )
    print(f"#Datapoints on train dataset: {len(dset['train'])}")
    print(f"#Datapoints on valid dataset: {len(dset['valid'])}")
    print(f"#Datapoints on test dataset: {len(dset['test'])}")

    testset_df = pd.DataFrame(dset["test"])
    if args['model_type'] == "copy":
        list_label = testset_df['tgt_text'].tolist()
        list_hyp = testset_df['ind_text'].tolist()
    elif args['model_type'] == "word-substitution":
        list_label = testset_df['tgt_text'].tolist()
        list_src = testset_df['ind_text'].tolist()
        list_hyp = translate_lexical_panlex(
            sentences=list_src,
            src_lang='ind',
            dst_lang=args["lang"]
        )
    elif args['model_type'] == "pbsmt":
        raise Error("Not Implemented")
    else:
        raise ValueError(f"Error: Unknown model_type {args['model_type']}")

    # Evaluation
    print("=========== EVALUATION PHASE ===========")
    test_loss, test_metrics, test_hyp, test_label = evaluate_classical(
        list_hyp=list_hyp, 
        list_label=list_label, 
        metrics_fn=args['metrics_fn'], 
    )

    metrics_scores.append(test_metrics)
    result_dfs.append(pd.DataFrame({
        'hyp': test_hyp, 
        'label': test_label
    }))
    
    result_df = pd.concat(result_dfs)
    metric_df = pd.DataFrame.from_records(metrics_scores)
    
    print('== Prediction Result ==')
    print(result_df.head())
    print()
    
    print('== Model Performance ==')
    print(metric_df)
    
    result_df.to_csv(output_dir + "/prediction_result.csv")
    metric_df.to_csv(output_dir + "/evaluation_result.csv")


if __name__ == "__main__":
    # Parse args
    args = get_generation_parser()
    args = append_generation_dataset_args(args)
    args = append_generation_model_args(args)

    if args['model_type'] in ["copy", "word-substitution"]:
        process_classical_benchmark(args)
    else:
        raise ValueError(f"Error: unrecognized classic benchmark: {args['model_type']}")