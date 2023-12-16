import os, sys
import csv
from os.path import exists
import glob

from numpy import argmax
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score
from prompts import get_prompt

import torch
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM

from sacremoses import MosesTokenizer
import datasets
from anyascii import anyascii
import time

from utils.data_utils import load_sequence_classification_dataset, SequenceClassificationDataset, load_dataset
from utils.metrics import sentiment_metrics_fn, generation_metrics_fn

DEBUG=True

"""# Loading NLG Datasets"""

lang_map = {
    'abs': 'Ambonese',
    'btk': 'Batak',
    'bew': 'Betawi',
    'bhp': 'Bima',
    'jav': 'Javanese',
    'mad': 'Madurese',
    'mak': 'Makassarese',
    'min': 'Minangkabau',
    'mui': 'Musi',
    'rej': 'Rejang',
    'sun': 'Sundanese',
    'ind': 'Indonesian'
}

def to_prompt(input, prompt, lang, to_ind=False, with_label=False):
    if not to_ind:
        src_text, tgt_text = input['ind_text'], input['tgt_text']
        src_lang, tgt_lang = lang_map['ind'], lang_map[lang]
    else:
        src_text, tgt_text = input['tgt_text'], input['ind_text']
        src_lang, tgt_lang = lang_map[lang], lang_map['ind']
        
    prompt = prompt.replace('[INPUT]', src_text)
    prompt = prompt.replace('[SOURCE]', src_lang).replace('[TARGET]', tgt_lang)    
    if with_label:
        prompt += " " + tgt_text
    return prompt

def load_nlg_tasks():
    meta = []
    for path in glob.glob('./data/*.csv'):
        meta.append(tuple(path.split('/')[-1][:-4].split('-')[:3]))
    meta = sorted(list(set(filter(lambda x: x[1] == 'mt', meta))))
    return { (dataset, task, lang) : load_dataset(dataset, task, lang) for (dataset, task, lang) in meta } 

def predict_generation(prompt, model_name):
    if "gpt" in model_name or "text" in model_name:
        return predict_generation_gpt(prompt, model_name)

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to('cuda')
    input_ids = inputs["input_ids"]
    input_size = inputs["input_ids"].shape[1]
    if 'mt0' in model_name:
        outputs = model.generate(**inputs, do_sample=True, 
             min_length=1, max_length=100)
        return tokenizer.decode(outputs[0], skip_special_tokens=True)
    else:
        outputs = model.generate(**inputs, do_sample=True, 
             min_length=input_size+1, max_length=input_size+100)
        return tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


if __name__ == '__main__':
    MODEL='bigscience/bloomz-3b'

    os.makedirs('./outputs_nlg', exist_ok=True) 

    if len(sys.argv) != 2:
        raise ValueError('main_nlg_prompt.py <model_path_or_name>')

    MODEL = sys.argv[1]

    # Load prompt
    prompt_templates = get_prompt()

    # Load Dataset
    print('Load NLG Datasets...')
    nlg_datasets = load_nlg_tasks()

    print(f'Loaded {len(nlg_datasets)} NLG datasets')
    for i, dset_subset in enumerate(nlg_datasets.keys()):
        print(f'{i} {dset_subset}')

    # Load Model
    tokenizer = AutoTokenizer.from_pretrained(MODEL, truncation_side='left') if ("gpt" not in MODEL and "text" not in MODEL) else None
    if "gpt" in MODEL or "text" in MODEL:
        model = None
    elif 'mt0' in MODEL:
        if "xxl" in MODEL:
            model = AutoModelForSeq2SeqLM.from_pretrained(MODEL, device_map="auto", load_in_8bit=True)
        else:
            model = AutoModelForSeq2SeqLM.from_pretrained(MODEL)
            model = model.to('cuda')
    else:
        model = AutoModelForCausalLM.from_pretrained(MODEL)
        model = model.to('cuda')
    if model is not None:
        model = model.eval()

    # Result Buffer
    metrics = { 'dataset':[], 'task':[], 'src_lang':[], 'tgt_lang':[] }
    
    ######
    # ind -> xxx
    ######
    for (dataset, task, lang), dset in nlg_datasets.items():        
        print(f'{dataset} | {task} | {lang}')
        if task not in prompt_templates or prompt_templates[task] is None:
            print('SKIP')
            continue
    
        # take test data
        data = dset['test']
      
        for prompt_id, prompt_template in enumerate(prompt_templates[task]):
            inputs = []
            preds = []
            golds = []  

            print(f"PROMPT ID: {prompt_id}")
            print(f"SAMPLE PROMPT: {to_prompt(data[0], prompt_template, lang, to_ind=False)}")
            
            # zero-shot inference
            if exists(f'./outputs_nlg/{dataset}_{task}_{prompt_id}_ind_{lang}_{MODEL.split("/")[-1]}.csv'):        
                print("Output exist, use existing log instead")
                with open(f'./outputs_nlg/{dataset}_{task}_{prompt_id}_ind_{lang}_{MODEL.split("/")[-1]}.csv') as csvfile:
                    reader = csv.DictReader(csvfile)
                    for row in reader:
                        inputs.append(row["Input"])
                        preds.append(row["Pred"])
                        golds.append(row["Gold"])
                print(f"Skipping until {len(preds)}")
            # if incomplete, continue
            if len(preds) < len(data):
                with torch.inference_mode():
                    for e, sample in enumerate(tqdm(data)):
                        if e < len(preds):
                            continue
                        prompt_text = to_prompt(sample, prompt_template, lang, to_ind=False)
                        pred = predict_generation(prompt_text, MODEL)
                        
                        inputs.append(prompt_text)
                        preds.append(pred)
                        golds.append(sample['tgt_text'])
                
                        # partial saving
                        if len(preds) % 10 == 0:
                            inference_df = pd.DataFrame(list(zip(inputs, preds, golds)), columns=['Input', 'Pred', 'Gold'])
                            inference_df.to_csv(f'./outputs_nlg/{dataset}_{task}_{prompt_id}_ind_{lang}_{MODEL.split("/")[-1]}.csv', index=False)
                        break

            # final save
            inference_df = pd.DataFrame(list(zip(inputs, preds, golds)), columns=['Input', 'Pred', 'Gold'])
            inference_df.to_csv(f'./outputs_nlg/{dataset}_{task}_{prompt_id}_ind_{lang}_{MODEL.split("/")[-1]}.csv', index=False)

            eval_metric = generation_metrics_fn(preds, golds)

            print(f'== {dataset}_{task}_ind_{lang} == ')
            for k, v in eval_metric.items():
                print(k, v)            
            print("===\n\n")
            eval_metric['prompt_id'] = prompt_id

            metrics['dataset'].append(dataset)
            metrics['task'].append(task)
            metrics['src_lang'].append('ind')
            metrics['tgt_lang'].append(lang)
            for k in eval_metric:
                if k not in metrics:
                    metrics[k] = []
                metrics[k].append(eval_metric[k])

    ######
    # xxx -> ind
    ######
    for (dataset, task, lang), dset in nlg_datasets.items():        
        print(f'{dataset} | {task} | {lang}')
        if task not in prompt_templates or prompt_templates[task] is None:
            print('SKIP')
            continue
    
        # take test data
        data = dset['test']
      
        for prompt_id, prompt_template in enumerate(prompt_templates[task]):
            inputs = []
            preds = []
            golds = []  

            print(f"PROMPT ID: {prompt_id}")
            print(f"SAMPLE PROMPT: {to_prompt(data[0], prompt_template, lang, to_ind=True)}")
            
            # zero-shot inference
            if exists(f'./outputs_nlg/{dataset}_{task}_{prompt_id}_{lang}_ind_{MODEL.split("/")[-1]}.csv'):        
                print("Output exist, use existing log instead")
                with open(f'./outputs_nlg/{dataset}_{task}_{prompt_id}_{lang}_ind_{MODEL.split("/")[-1]}.csv') as csvfile:
                    reader = csv.DictReader(csvfile)
                    for row in reader:
                        inputs.append(row["Input"])
                        preds.append(row["Pred"])
                        golds.append(row["Gold"])
                print(f"Skipping until {len(preds)}")
            # if incomplete, continue
            if len(preds) < len(data):
                with torch.inference_mode():
                    for e, sample in enumerate(tqdm(data)):
                        if e < len(preds):
                            continue
                        prompt_text = to_prompt(sample, prompt_template, lang, to_ind=True)
                        pred = predict_generation(prompt_text, MODEL)
                        
                        inputs.append(prompt_text)
                        preds.append(pred)
                        golds.append(sample['ind_text'])
                
                        # partial saving
                        if len(preds) % 10 == 0:
                            inference_df = pd.DataFrame(list(zip(inputs, preds, golds)), columns=['Input', 'Pred', 'Gold'])
                            inference_df.to_csv(f'./outputs_nlg/{dataset}_{task}_{prompt_id}_{lang}_ind_{MODEL.split("/")[-1]}.csv', index=False)

            # final save
            inference_df = pd.DataFrame(list(zip(inputs, preds, golds)), columns=['Input', 'Pred',  'Gold'])
            inference_df.to_csv(f'./outputs_nlg/{dataset}_{task}_{prompt_id}_{lang}_ind_{MODEL.split("/")[-1]}.csv', index=False)

            eval_metric = generation_metrics_fn(preds, golds)
            print(f'== {dataset}_{task}_{lang}_ind == ')
            for k, v in eval_metric.items():
                print(k, v)            
            print("===\n\n")
            eval_metric['prompt_id'] = prompt_id

            metrics['dataset'].append(dataset)
            metrics['task'].append(task)
            metrics['src_lang'].append(lang)
            metrics['tgt_lang'].append('ind')
            for k in eval_metric:
                if k not in metrics:
                    metrics[k] = []
                metrics[k].append(eval_metric[k])

    pd.DataFrame.from_dict(metrics).reset_index().to_csv(f'./outputs_nlg/nlg_results_{MODEL.split("/")[-1]}.csv', index=False)
