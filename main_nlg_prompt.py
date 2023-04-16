import os, sys
import csv
from os.path import exists

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
import openai
import time
from retry import retry

openai.api_key = ""


DEBUG=True

CONFIG_NAMES = ['mt_enghinglish']

""" Generation metrics """
bleu = datasets.load_metric('bleu')
rouge = datasets.load_metric('rouge')
sacrebleu = datasets.load_metric('sacrebleu')
chrf = datasets.load_metric('chrf')
squad_v2_metric = datasets.load_metric('squad_v2')
mt = MosesTokenizer(lang='id')
"""# Loading NLG Datasets"""

def generation_metrics_fn(list_hyp, list_label):
    # hyp and label are both list of string
    list_hyp_bleu = list(map(lambda x: mt.tokenize(x), list_hyp))
    list_label_bleu = list(map(lambda x: [mt.tokenize(x)], list_label))    
    list_label_sacrebleu = list(map(lambda x: [x], list_label))
    
    metrics = {}
    metrics["BLEU"] = bleu._compute(list_hyp_bleu, list_label_bleu)['bleu'] * 100
    metrics["SacreBLEU"] = sacrebleu._compute(list_hyp, list_label_sacrebleu)['score']
    metrics["chrF++"] = chrf._compute(list_hyp, list_label_sacrebleu)['score']
    
    rouge_score = rouge._compute(list_hyp, list_label)
    metrics["ROUGE1"] = rouge_score['rouge1'].mid.fmeasure * 100
    metrics["ROUGE2"] = rouge_score['rouge2'].mid.fmeasure * 100
    metrics["ROUGEL"] = rouge_score['rougeL'].mid.fmeasure * 100
    metrics["ROUGELsum"] = rouge_score['rougeLsum'].mid.fmeasure * 100
    return metrics

lang_map = {
    'hi': 'hindi',
    'en': 'english',
}
def to_prompt(input, prompt, src_lang, tgt_lang, with_label=False):
    prompt = prompt.replace('[INPUT]', input['translation'][src_lang])
    prompt = prompt.replace('[SOURCE]', lang_map[src_lang]).replace('[TARGET]', lang_map[tgt_lang])
    
    if with_label:
        prompt += " " + input['translation'][tgt_lang]
    return prompt

def load_nlg_tasks():
    dsets = {}
    for config in CONFIG_NAMES: #TODO: Add some NLG Tasks here
        dsets[config] = datasets.load_dataset(
            'CodeSwitchBenchmark/CodeMixBench_MT', config,
            use_auth_token='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'
        )
    return dsets

# they sometimes timeout
@retry(Exception, tries=5, delay=1)
def predict_generation_gpt(prompt, model_name):
    if "turbo" in model_name:
        response = openai.ChatCompletion.create(
          model=model_name,
          messages=[
                {"role": "user", "content": prompt},
            ]
        )

        return response['choices'][0]['message']['content'].strip()
    else:
        response = openai.Completion.create(
            model=model_name,
            prompt=prompt,
            max_tokens=200,
          )
        return response['choices'][0]['text'].strip()

def predict_generation(prompt, model_name):
    if "gpt" in model_name or "text" in model_name:
        return predict_generation_gpt(prompt, model_name)

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to('cuda')
    input_ids = inputs["input_ids"]
    input_size = inputs["input_ids"].shape[1]
    outputs = model.generate(**inputs, do_sample=True, 
             min_length=input_size+1, max_length=input_size+100)
    if 'mt0' in model_name:
        return tokenizer.decode(outputs[0], skip_special_tokens=True)
    else:
        return tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


if __name__ == '__main__':
    MODEL='bigscience/bloomz-3b'

    os.makedirs('./outputs_nlg', exist_ok=True) 

    if len(sys.argv) != 5:
        raise ValueError('main_nlg_prompt.py <model_path_or_name> <src_lang> <tgt_lang> <n_shot>')

    MODEL = sys.argv[1]
    src_lang = sys.argv[2]
    tgt_lang = sys.argv[3]
    N_SHOT = int(sys.argv[4])

    # Load prompt
    prompt_templates = get_prompt(CONFIG_NAMES)

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
            model = AutoModelForSeq2SeqLM.from_pretrained(MODEL, device_map="auto")
        else:
            model = AutoModelForSeq2SeqLM.from_pretrained(MODEL)
            model = model.to('cuda')
    else:
        model = AutoModelForCausalLM.from_pretrained(MODEL)
        model = model.to('cuda')
    if model is not None:
        model = model.eval()

    metrics = { 'dataset':[]}
    for i, dset_subset in enumerate(nlg_datasets.keys()):
        if dset_subset not in prompt_templates or prompt_templates[dset_subset] is None:
            continue

        data = nlg_datasets[dset_subset]['validation']
        few_shot_data = nlg_datasets[dset_subset]['train']

      
        for prompt_id, prompt_template in enumerate(prompt_templates[dset_subset]):
            inputs = []
            preds = []
            preds_latin = []
            golds = []  
            print(f"PROMPT ID: {prompt_id}")
            print(f"SAMPLE PROMPT: {to_prompt(data[0], prompt_template, src_lang, tgt_lang)}")

            few_shot_text_list = []
            if N_SHOT > 0:
                for sample in tqdm(few_shot_data):
                    # skip shot examples
                    if len(sample['translation'][src_lang]) < 20:
                        continue
                    few_shot_text_list.append(
                        to_prompt(sample, prompt_template, src_lang, tgt_lang, with_label=True)
                    )
                    if len(few_shot_text_list) == N_SHOT:
                        break
            print(f'FEW SHOT SAMPLES: {few_shot_text_list}')
            
            # zero-shot inference
            if exists(f'./outputs_nlg/{dset_subset}_{prompt_id}_{N_SHOT}_{src_lang}_{tgt_lang}_{MODEL.split("/")[-1]}.csv'):        
                print("Output exist, use existing log instead")
                with open(f'./outputs_nlg/{dset_subset}_{prompt_id}_{N_SHOT}_{src_lang}_{tgt_lang}_{MODEL.split("/")[-1]}.csv') as csvfile:
                    reader = csv.DictReader(csvfile)
                    for row in reader:
                        inputs.append(row["Input"])
                        preds.append(row["Pred"])
                        preds_latin.append(row["Pred_Latin"])
                        golds.append(row["Gold"])
                print(f"Skipping until {len(preds)}")
            # if incomplete, continue
            if len(preds) < len(data):
                with torch.inference_mode():
                    for e, sample in enumerate(tqdm(data)):
                        if e < len(preds):
                            continue
                        prompt_text = to_prompt(sample, prompt_template, src_lang, tgt_lang)
                        prompt_text = '\n\n'.join(few_shot_text_list + [prompt_text])
                        pred = predict_generation(prompt_text, MODEL)
                        inputs.append(prompt_text)
                        preds.append(pred)
                        preds_latin.append(anyascii(pred))
                        golds.append(sample['translation'][tgt_lang])
                        #print('pred:', pred)
                        #print('gold:', sample['translation'][tgt_lang])
                        #print()
                
                        # partial saving
                        if len(preds) % 10 == 0:
                            inference_df = pd.DataFrame(list(zip(inputs, preds, preds_latin, golds)), columns=['Input', 'Pred', 'Pred_Latin', 'Gold'])
                            inference_df.to_csv(f'./outputs_nlg/{dset_subset}_{prompt_id}_{N_SHOT}_{src_lang}_{tgt_lang}_{MODEL.split("/")[-1]}.csv', index=False)

            # final save
            inference_df = pd.DataFrame(list(zip(inputs, preds, preds_latin, golds)), columns=['Input', 'Pred', 'Pred_Latin', 'Gold'])
            inference_df.to_csv(f'./outputs_nlg/{dset_subset}_{prompt_id}_{N_SHOT}_{src_lang}_{tgt_lang}_{MODEL.split("/")[-1]}.csv', index=False)

            # to accomodate old bug where list are not properly re-initiated
            inputs = inputs[-len(data):]
            preds = preds[-len(data):]
            preds_latin = preds_latin[-len(data):]
            golds = golds[-len(data):]

            eval_metric = generation_metrics_fn(preds, golds)
            eval_metric_latin = generation_metrics_fn(preds_latin, golds)
            for key, value in eval_metric_latin.items():
                eval_metric[f'{key}_latin'] = value

            print(f'== {dset_subset} == ')
            for k, v in eval_metric.items():
                print(k, v)            
            print("===\n\n")
            eval_metric['prompt_id'] = prompt_id

            metrics['dataset'].append(dset_subset)
            for k in eval_metric:
                if k not in metrics:
                    metrics[k] = []
                metrics[k].append(eval_metric[k])


    pd.DataFrame.from_dict(metrics).reset_index().to_csv(f'./outputs_nlg/nlg_results_{N_SHOT}_{src_lang}_{tgt_lang}_{MODEL.split("/")[-1]}.csv', index=False)
