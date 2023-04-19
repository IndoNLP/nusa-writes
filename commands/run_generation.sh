#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

# hyperparams to execute script, pls change it appropriately for your needs
num_sample=-1
train_batch_size=8
valid_batch_size=8
test_batch_size=8
seed=43
n_epochs=100
lr=1e-5
device=cuda

# echo "Current directory: $PWD"

# ## -- NUSA_KALIMAT -- ##
# # mt w/o models checkpoint (for indo-bart and indo-gpt)
# model_choices=(indo-bart indo-gpt2)
# for lang in abs bew bhp btk jav mad mak min mui rej sun; do
#     for ((i = 0; i < ${#model_choices[@]}; ++i)); do
#         echo "Executing MT task on lang ${lang} and model choice ${model_choices[$i]}"
#         python main_generation.py --lang $lang --model_type ${model_choices[$i]} --device $device --n_epochs $n_epochs --lr $lr --train_batch_size $train_batch_size --valid_batch_size $valid_batch_size --test_batch_size $test_batch_size --seed $seed --num_sample $num_sample --force
#         # CUDA_VISIBLE_DEVICES=0 python main_generation.py --lang $lang --model_type ${model_choices[$i]} --device $device --n_epochs $n_epochs --lr $lr --train_batch_size $train_batch_size --valid_batch_size $valid_batch_size --test_batch_size $test_batch_size --seed $seed --num_sample $num_sample &
#     done
#     wait
#     rm -r save/nusa_kalimat/mt/$lang/*/checkpoint*
# done


# # mt w/ models checkpoint (for baseline-mbart and baseline-mt5)
# model_checkpoints=(facebook/mbart-large-50-one-to-many-mmt google/mt5-base)
# model_choices_w_checkpoint=(baseline-mbart baseline-mt5)
# for lang in abs bew bhp btk jav mad mak min mui rej sun; do
#     for ((i = 0; i < ${#model_choices_w_checkpoint[@]}; ++i)); do
#         echo "Executing MT task on lang ${lang} and model choice ${model_choices[$i]}"
#         python main_generation.py --lang $lang --model_type ${model_choices_w_checkpoint[$i]} --model_checkpoint ${model_checkpoints[$i]} --device $device --n_epochs $n_epochs --lr $lr --train_batch_size $train_batch_size --valid_batch_size $valid_batch_size --test_batch_size $test_batch_size --seed $seed --num_sample $num_sample --force
#         # CUDA_VISIBLE_DEVICES=0 python main_generation.py --lang $lang --model_type ${model_choices_w_checkpoint[$i]} --model_checkpoint ${model_checkpoints[$i]} --device $device --n_epochs $n_epochs --lr $lr --train_batch_size $train_batch_size --valid_batch_size $valid_batch_size --test_batch_size $test_batch_size --seed $seed --num_sample $num_sample --force
#     done
#     wait
#     rm -r save/nusa_kalimat/mt/$lang/*/checkpoint*
# done
