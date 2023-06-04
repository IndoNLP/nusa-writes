#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for causal language modeling (GPT, GPT-2, CTRL, ...) on a text file or a dataset.

Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=text-generation
"""
# You can also adapt this script on your own causal language modeling task. Pointers for this are left as comments.

import string
from collections import Counter
import logging
import math
import os
import sys
from dataclasses import dataclass, field
from itertools import chain
from typing import Optional

import glob
import pandas as pd
import numpy as np

import datasets
import torch
from datasets import load_dataset
from nusacrowd import NusantaraConfigHelper
from modules.tokenization_indonlg import IndoNLGTokenizer

import transformers
from transformers import (
    GPT2Config,
    GPT2LMHeadModel,
    CONFIG_MAPPING,
    MODEL_FOR_CAUSAL_LM_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    is_torch_tpu_available,
    set_seed,
    DataCollatorForLanguageModeling
)
from transformers.testing_utils import CaptureLogger
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version


logger = logging.getLogger(__name__)

MODEL_CONFIG_CLASSES = list(MODEL_FOR_CAUSAL_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization.Don't set if you want to train a model from scratch."
            )
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_overrides: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override some existing default config settings when a model is trained from scratch. Example: "
                "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"
            )
        },
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `huggingface-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )
    torch_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the "
                "dtype will be automatically derived from the model's weights."
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )
    low_cpu_mem_usage: bool = field(
        default=False,
        metadata={
            "help": (
                "It is an option to create the model as an empty shell, then only materialize its parameters when the pretrained weights are loaded."
                "set True will benefit LLM loading time and RAM consumption."
            )
        },
    )

    def __post_init__(self):
        if self.config_overrides is not None and (self.config_name is not None or self.model_name_or_path is not None):
            raise ValueError(
                "--config_overrides can't be used in combination with --config_name or --model_name_or_path"
            )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_lang: Optional[str] = field(
        default=None, metadata={"help": "The language of the dataset to be used)."}
    )
    rebalance: bool = field(
        default=False, metadata={"help": "Whether to rebalance the data size or not"}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    streaming: bool = field(default=False, metadata={"help": "Enable streaming mode"})
    block_size: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Optional input sequence length after tokenization. "
                "The training dataset will be truncated in block of this size for training. "
                "Default to the model max input length for single sentence inputs (take into account special tokens)."
            )
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    keep_linebreaks: bool = field(
        default=True, metadata={"help": "Whether to keep line breaks when using TXT files or not."}
    )

    def __post_init__(self):
        if self.streaming:
            require_version("datasets>=2.0.0", "The streaming feature requires `datasets>=2.0.0`")

        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json", "txt"], "`train_file` should be a csv, a json or a txt file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in ["csv", "json", "txt"], "`validation_file` should be a csv, a json or a txt file."


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    send_example_telemetry("run_clm", model_args, data_args)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
    # 'text' is found. You can easily tweak this behavior (see below).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    # Downloading and loading a dataset from the hub.   
    
    ###
    # Load Datasets
    ###
    lang = data_args.dataset_lang
    
    # Load Wiki
    wiki_dset = {}
    if lang == 'sun':
        wiki_lang = 'su'
    elif lang == 'jav':
        wiki_lang = 'jv'
    else:
        wiki_lang = lang
    wiki_dset = datasets.load_dataset('olm/wikipedia', language=wiki_lang, date="20221220")['train'].to_pandas()
        
    # Load Paragraph
    dfs = []
    for path in glob.glob(f'./data/nusa_alinea-paragraph-{lang}-*.csv'):
        _, _, lang, _ = path[:-4].split('/')[-1].split('-')
        dfs.append(pd.read_csv(path))
    para_dset = pd.concat(dfs)

    # Load MT
    if lang != 'bug':
        dfs = []
        for path in glob.glob(f'./data/nusa_kalimat-mt-{lang}-*.csv'):
            _, _, lang, _ = path[:-4].split('/')[-1].split('-')
            dfs.append(pd.read_csv(path))
        mt_dset = pd.concat(dfs)
        mt_dset['text'] = mt_dset['tgt_text']
    else:
        mt_dset = None
    
    # Load NusaX Text
    conhelps = NusantaraConfigHelper()
    nusax_dset = conhelps.filtered(lambda x: x.config.name == f'nusax_senti_{lang}_nusantara_text')[0].load_dataset()
    nusax_test_dset = datasets.Dataset.from_dict({'text': nusax_dset['train']['text'] + nusax_dset['validation']['text'] + nusax_dset['test']['text']})
    
    ###
    # Chunk & Count Text
    ###
    replacement_rules = str.maketrans('', '', string.punctuation)
    wiki_dset['clean_text'] = wiki_dset['text'].apply(lambda x: x.lower().translate(replacement_rules).replace('\n',' '))
    para_dset['clean_text'] = para_dset['text'].apply(lambda x: x.lower().translate(replacement_rules).replace('\n',' '))
    if mt_dset is not None:
        mt_dset['clean_text'] = mt_dset['text'].apply(lambda x: x.lower().translate(replacement_rules).replace('\n',' '))
    
    wiki_dset['token_count'] = wiki_dset['clean_text'].apply(lambda x: len(x.split(' ')))
    para_dset['token_count'] = para_dset['clean_text'].apply(lambda x: len(x.split(' ')))
    if mt_dset is not None:
        mt_dset['token_count'] = mt_dset['clean_text'].apply(lambda x: len(x.split(' ')))
        
    print('== Token Size ==')
    print(f'{lang} WIKI: ', wiki_dset['token_count'].sum())
    print(f'{lang} PARA: ', para_dset['token_count'].sum())
    if mt_dset is not None:
        print(f'{lang} MT: ', mt_dset['token_count'].sum())
    
    # Rebalancing
    if data_args.rebalance:
        token_sizes = { 'bug': 118392, 'mad': 106335, 'jav': 208034, 'min': 211084, 'sun': 209492 }
        token_size = token_sizes[lang]
        
        ###
        # shuffle & resample
        ###
        wiki_dset = wiki_dset.sample(n=len(wiki_dset), random_state=training_args.seed)
        n_tok = 0
        for i,c in enumerate(wiki_dset['token_count']):
            n_tok += c
            if n_tok >= token_size:
                break
        wiki_dset = wiki_dset.iloc[:i+1]
        
        para_dset = para_dset.sample(n=len(para_dset), random_state=training_args.seed)
        n_tok = 0
        for i,c in enumerate(para_dset['token_count']):
            n_tok += c
            if n_tok >= token_size:
                break
        para_dset = para_dset.iloc[:i+1]
        
        if mt_dset is not None:
            mt_dset = mt_dset.sample(n=len(mt_dset), random_state=training_args.seed)
            n_tok = 0
            for i,c in enumerate(mt_dset['token_count']):
                n_tok += c
                if n_tok >= token_size:
                    break
            mt_dset = mt_dset.iloc[:i+1]
        
        print('== Rebalanced Token Size ==')
        print(f'{lang} WIKI: ', wiki_dset['token_count'].sum())
        print(f'{lang} PARA: ', para_dset['token_count'].sum())
        if mt_dset is not None:
            print(f'{lang} MT: ', mt_dset['token_count'].sum())

    ###
    # Split Dataset
    ###
    wiki_dset = datasets.Dataset.from_pandas(wiki_dset[['text']])
    para_dset = datasets.Dataset.from_pandas(para_dset[['text']])
    if mt_dset is not None:
        mt_dset = datasets.Dataset.from_pandas(mt_dset[['text']]) 

    ###
    # Select Train & Validation Dataset
    #   this is kinda the stupid way, should've filter the data before preprocessing and not after :p
    ###
    if data_args.dataset_name == 'wikipedia':
        train_dset = wiki_dset 
    elif data_args.dataset_name == 'paragraph':
        train_dset = para_dset 
    elif data_args.dataset_name == 'translation':
        train_dset = mt_dset 
    else:
        raise ValueError(f'Invalid `data_args.dataset_name`: "{data_args.dataset_name}"')
    
    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    tokenizer = IndoNLGTokenizer.from_pretrained('indobenchmark/indogpt')
    config = GPT2Config.from_pretrained('indobenchmark/indogpt')
    config.n_embd = 128
    config.n_head = 2
    config.n_layer = 2
    config.n_positions = 256
    config.n_ctx = 256
    
    model = GPT2LMHeadModel(config=config)
    n_params = sum({p.data_ptr(): p.numel() for p in model.parameters()}.values())
    logger.info(f"Training new model from scratch - Total size={n_params/2**20:.2f}M params")
    
    # Preprocessing the datasets.
    # First we tokenize all the texts.
    # since this will be pickled to avoid _LazyModule error in Hasher force logger loading before tokenize_function
    tok_logger = transformers.utils.logging.get_logger("transformers.tokenization_utils_base")

    def tokenize_function(examples):
        with CaptureLogger(tok_logger) as cl:
            output = tokenizer(examples[text_column_name], truncation=True, max_length=128)
        # clm input could be much much longer than block_size
        if "Token indices sequence length is longer than the" in cl.out:
            tok_logger.warning(
                "^^^^^^^^^^^^^^^^ Please ignore the warning above - this long input will be chunked into smaller bits"
                " before being passed to the model."
            )
        return output
    
    text_column_name = 'text'
    with training_args.main_process_first(desc="dataset map tokenization"):
        train_dset = train_dset.map(
            tokenize_function,
            batched=False,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=list(train_dset.features),
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on dataset",
        )
        nusax_test_dset = nusax_test_dset.map(
            tokenize_function,
            batched=False,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=list(nusax_test_dset.features),
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on dataset",
        )

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dset if training_args.do_train else None,
        eval_dataset=nusax_test_dset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator= DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics

        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate()

        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(nusax_test_dset)
        metrics["eval_samples"] = min(max_eval_samples, len(nusax_test_dset))
        try:
            perplexity = math.exp(metrics["eval_loss"])
        except OverflowError:
            perplexity = float("inf")
        metrics["perplexity"] = perplexity

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)



def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()