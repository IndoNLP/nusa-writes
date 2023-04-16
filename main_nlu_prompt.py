import os, sys
import csv
from os.path import exists

from numpy import argmax
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score
from prompts import get_prompt
import datasets

import torch
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
from nusacrowd import NusantaraConfigHelper

DEBUG=False
CONFIG_NAMES = ['sentimix_spaeng', 'tamil_mixsentiment', 'malayalam_mixsentiment']

"""# Loading NLU Datasets"""
def to_prompt(input, prompt, labels, with_label=False):
    # single label
    if 'text' in input:
        prompt = prompt.replace('[INPUT]', input['text'])
    else:
        prompt = prompt.replace('[INPUT_A]', input['text_1'])
        prompt = prompt.replace('[INPUT_B]', input['text_2'])

    # replace [OPTIONS] to A, B, or C
    if "[OPTIONS]" in prompt:
        new_labels = [f'{"or " if i == len(labels) - 1 else ""}{l}' for i, l in enumerate(labels)]
        if len(new_labels) > 2:
            prompt = prompt.replace('[OPTIONS]', ', '.join(new_labels))
        else:
            prompt = prompt.replace('[OPTIONS]', ' '.join(new_labels))

    if with_label:
        prompt = prompt.replace('[LABELS_CHOICE]', labels[input['label']])
        
    return prompt

def load_nlu_tasks():
    dsets = {}
    for config in CONFIG_NAMES:
        dsets[config] = datasets.load_dataset(
            'CodeSwitchBenchmark/CodeMixBench', config,
            use_auth_token='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'
        )
    return dsets

@torch.no_grad()
def get_logprobs(model, tokenizer, prompt, label_ids=None, label_attn=None):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to('cuda')
    input_ids, output_ids = inputs["input_ids"], inputs["input_ids"][:, 1:]
    outputs = model(**inputs, labels=input_ids)
    logits = outputs.logits
    
    if model.config.is_encoder_decoder:
        logprobs = torch.gather(F.log_softmax(logits, dim=2), 2, label_ids.unsqueeze(2)) * label_attn.unsqueeze(2)
        return logprobs.sum() / label_attn.sum()
    else:
        logprobs = torch.gather(F.log_softmax(logits, dim=2), 2, output_ids.unsqueeze(2))
        return logprobs.mean()

def predict_classification(model, tokenizer, prompt, labels):
    if model.config.is_encoder_decoder:
        labels_encoded = tokenizer(labels, add_special_tokens=False, padding=True, return_tensors='pt')
        list_label_ids =labels_encoded['input_ids'].to('cuda')
        list_label_attn =labels_encoded['attention_mask'].to('cuda')
        probs = [
                    get_logprobs(model, tokenizer, prompt.replace('[LABELS_CHOICE]', ''), label_ids.view(1,-1), label_attn.view(1,-1)) 
                     for (label_ids, label_attn) in zip(list_label_ids, list_label_attn)
                ]
    else:
        probs = [get_logprobs(model, tokenizer, prompt.replace('[LABELS_CHOICE]', label)) for label in labels]
    return probs

if __name__ == '__main__':
    if len(sys.argv) != 3:
        raise ValueError('main_nlu_prompt.py <model_path_or_name> <n_shot>')

    MODEL = sys.argv[1]
    K_SHOT = int(sys.argv[2])

    os.makedirs('./outputs_nlu', exist_ok=True) 

    # Load Prompt
    prompt_templates = get_prompt(CONFIG_NAMES)

    # Load Dataset
    print('Load NLU Datasets...')
    nlu_datasets = load_nlu_tasks()

    print(f'Loaded {len(nlu_datasets)} NLU datasets')
    for i, dset_subset in enumerate(nlu_datasets.keys()):
        print(f'{i} {dset_subset}')

    # Load Model
    tokenizer = AutoTokenizer.from_pretrained(MODEL, truncation_side='left')
    if "bloom" in MODEL or "xglm" in MODEL:
        model = AutoModelForCausalLM.from_pretrained(MODEL).to('cuda')
    else:
        if "xxl" not in MODEL:
            model = AutoModelForSeq2SeqLM.from_pretrained(MODEL).to('cuda')
        else:
             model = AutoModelForSeq2SeqLM.from_pretrained(MODEL, device_map="auto")

    model.eval()
    torch.no_grad()

    metrics = { 'dataset':[], 'prompt_id':[], 'accuracy':[], 'macro_f1':[], 'micro_f1':[] }
    for i, dset_subset in enumerate(nlu_datasets.keys()):
        print(f'{i} {dset_subset}')
        if dset_subset not in prompt_templates or prompt_templates[dset_subset] is None:
            print('SKIP')
            continue

        data = nlu_datasets[dset_subset]['test']
        few_shot_data = nlu_datasets[dset_subset]['train']

        # preprocess label (lower case & translate)
        label_names = data.features['label'].names
        label_names = [str(label).lower().replace("_"," ") for label in label_names]
        
        # sample prompt
        print(f"LABEL NAME: {label_names}")

        for prompt_id, prompt_template in enumerate(prompt_templates[dset_subset]):
            inputs = []
            preds = []
            golds = []
            
            print(f'prompt_id: {prompt_id}, k_shot: {K_SHOT}, model: {MODEL.split("/")[-1]}')
            print(f"SAMPLE PROMPT: {to_prompt(data[0], prompt_template, label_names)}")

            # inference
            if exists(f'outputs/{dset_subset}_{prompt_id}_{K_SHOT}_{MODEL.split("/")[-1]}.csv'):
                print("Output exist, use partial log instead")
                with open(f'outputs_nlu/{dset_subset}_{prompt_id}_{K_SHOT}_{MODEL.split("/")[-1]}.csv') as csvfile:
                    reader = csv.DictReader(csvfile)
                    for row in reader:
                        inputs.append(row["Input"])
                        preds.append(row["Pred"])
                        golds.append(row["Gold"])

                print(f"Skipping until {len(preds)}")

            # if incomplete, continue
            if len(preds) < len(data):
                few_shot_list = [[] for _ in range(len(label_names))]
                if K_SHOT > 0: # Take N-way K-shot examples
                    few_shot_done = [False for _ in range(len(label_names))]
                    for sample in few_shot_data:
                        if not few_shot_done[sample['label']]:
                            # Add to few shot example
                            few_shot_list[sample['label']].append(sample)
                            
                            # Check if already fulfill K-shot for the current label
                            if len(few_shot_list[sample['label']]) == K_SHOT:
                                # Flag the current label
                                few_shot_done[sample['label']] = True
                                
                                # Break if all labels are done
                                if sum(few_shot_done) == len(label_names):
                                    break
                
                # Take one sample per label until all K_SHOT is fulfilled
                few_shot_text_list = []
                for i in range(K_SHOT):
                    for j in range(len(label_names)):
                        few_shot_text_list.append(
                            to_prompt(few_shot_list[j].pop(), prompt_template, label_names, with_label=True)
                        )
                        
                print(f'FEW SHOT SAMPLES: {few_shot_text_list}')

                with torch.inference_mode():
                    for e, sample in enumerate(tqdm(data)):
                        if e < len(preds):
                            continue
                        # perform zero-shot / few-shot Inference
                        prompt_text = to_prompt(sample, prompt_template, label_names, with_label=False)
                        prompt_text = '\n\n'.join(few_shot_text_list + [prompt_text])
                        out = predict_classification(model, tokenizer, prompt_text, label_names)
                        pred = argmax([o.cpu().detach() for o in out])
                        inputs.append(prompt_text)
                        preds.append(pred)
                        golds.append(sample['label'])

                        # partial saving
                        if len(preds) % 10 == 0:
                            inference_df = pd.DataFrame(list(zip(inputs, preds, golds)), columns =["Input", 'Pred', 'Gold'])
                            inference_df.to_csv(f'outputs_nlu/{dset_subset}_{prompt_id}_{K_SHOT}_{MODEL.split("/")[-1]}.csv', index=False)

                inference_df = pd.DataFrame(list(zip(inputs, preds, golds)), columns =["Input", 'Pred', 'Gold'])
                inference_df.to_csv(f'outputs_nlu/{dset_subset}_{prompt_id}_{K_SHOT}_{MODEL.split("/")[-1]}.csv', index=False)
            # if output log exists, skip
            else:
                print("Output exist, use existing log instead")
                with open(f'outputs_nlu/{dset_subset}_{prompt_id}_{K_SHOT}_{MODEL.split("/")[-1]}.csv') as csvfile:
                    reader = csv.DictReader(csvfile)
                    for row in reader:
                        inputs.append(row["Input"])
                        preds.append(row["Pred"])
                        golds.append(row["Gold"])

            # to accomodate old bug where list are not properly re-initiated
            inputs = inputs[-len(data):]
            preds = preds[-len(data):]
            golds = golds[-len(data):]

            acc, macro_f1, micro_f1 = accuracy_score(golds, preds), f1_score(golds, preds, average='macro'), f1_score(golds, preds, average='micro')
            print(dset_subset)
            print('accuracy', acc)
            print('f1 macro', macro_f1)
            print('f1 micro', micro_f1)
            print("===\n\n")
            metrics['dataset'].append(dset_subset)
            metrics['prompt_id'].append(prompt_id)
            metrics['accuracy'].append(acc)
            metrics['macro_f1'].append(macro_f1)
            metrics['micro_f1'].append(micro_f1)

    pd.DataFrame.from_dict(metrics).reset_index().to_csv(f'outputs_nlu/nlu_results_{K_SHOT}_{MODEL.split("/")[-1]}.csv', index=False)
