#!/bin/bash

# ONLY FOR CHECKING PURPOSES

export CUDA_VISIBLE_DEVICES=0

##### RUN_NLU #####

python main.py --dataset_name nusa_kalimat --task emot --lang btk --model_checkpoint bert-base-multilingual-uncased --n_epochs 100 --lr 1e-5 --train_batch_size 32 --eval_batch_size 32 --seed 0 --num_sample -1 --force

python main.py --dataset_name nusa_kalimat --task emot --lang btk --model_checkpoint xlm-roberta-base --n_epochs 100 --lr 1e-5 --train_batch_size 32 --eval_batch_size 32 --seed 0 --num_sample -1 --force
rm -r save/nusa_kalimat/emot/btk/*/checkpoint*

python main.py --dataset_name nusa_kalimat --task senti --lang btk --model_checkpoint bert-base-multilingual-uncased --n_epochs 100 --lr 1e-5 --train_batch_size 32 --eval_batch_size 32 --seed 0 --num_sample -1 --force

python main.py --dataset_name nusa_kalimat --task senti --lang btk --model_checkpoint xlm-roberta-base --n_epochs 100 --lr 1e-5 --train_batch_size 32 --eval_batch_size 32 --seed 0 --num_sample -1 --force
rm -r save/nusa_kalimat/senti/btk/*/checkpoint*

##### RUN_NLG #####

python main_generation.py --lang rej --model_type indo-bart --device "cuda" --n_epochs 100 --lr 1e-5 --train_batch_size 8 --valid_batch_size 8 --test_batch_size 8 --seed 43 --num_sample -1 --force

python main_generation.py --lang rej --model_type baseline-mbart --model_checkpoint facebook/mbart-large-50-one-to-many-mmt --device "cuda" --n_epochs 100 --lr 1e-5 --train_batch_size 8 --valid_batch_size 8 --test_batch_size 8 --seed 43 --num_sample -1 --force

##### RUN_LOLO #####

python main_lolo.py --dataset_name nusa_kalimat --task senti --lang abs --model_checkpoint xlm-roberta-large --n_epochs 100 --lr 1e-5 --train_batch_size 16 --eval_batch_size 16 --grad_accum 1 --seed 43 --num_sample -1 --device cuda

python main_lolo.py --dataset_name nusa_kalimat --task senti --lang mui --model_checkpoint xlm-roberta-large --n_epochs 100 --lr 1e-5 --train_batch_size 16 --eval_batch_size 16 --grad_accum 1 --seed 43 --num_sample -1 --device cuda