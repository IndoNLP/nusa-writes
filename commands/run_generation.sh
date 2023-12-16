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

echo "Current directory: $PWD"

# -- NUSA_KALIMAT -- ##
model_choices=(indo-bart indo-gpt2)
# for lang in abs bew bhp btk jav mad mak min mui rej sun; do
for lang in min; do
    for ((i = 0; i < ${#model_choices[@]}; ++i)); do
        echo "Executing MT task on lang ${lang} and model choice ${model_choices[$i]}"
        CUDA_VISIBLE_DEVICES=$((i+5)) python main_generation.py --lang $lang --model_type ${model_choices[$i]} --device $device --n_epochs $n_epochs --lr 1e-4 --gamma 0.98 --train_batch_size 2 --valid_batch_size 2 --test_batch_size 2 --grad_accum 4 --seed $seed --num_sample $num_sample &
        # CUDA_VISIBLE_DEVICES=0 python main_generation.py --lang $lang --model_type ${model_choices[$i]} --device $device --n_epochs $n_epochs --lr $lr --train_batch_size $train_batch_size --valid_batch_size $valid_batch_size --test_batch_size $test_batch_size --seed $seed --num_sample $num_sample &
    done

    for ((i = 0; i < ${#model_choices[@]}; ++i)); do
        wait
    done
    
    rm -r save/nusa_kalimat/mt/$lang/*/checkpoint*
done

# mt w/ models checkpoint for baseline-mt5
model_checkpoints=(google/mt5-base)
model_choices_w_checkpoint=(baseline-mt5)
for lang in abs bew bhp btk jav mad mak min mui rej sun; do
    for ((i = 0; i < ${#model_choices_w_checkpoint[@]}; ++i)); do
        echo "Executing MT task on lang ${lang} and model choice ${model_choices[$i]}"
        CUDA_VISIBLE_DEVICES=0 python main_generation.py --lang $lang --model_type ${model_choices_w_checkpoint[$i]} --model_checkpoint ${model_checkpoints[$i]} --device $device --n_epochs $n_epochs --lr 5e-4 --gamma 0.98 --train_batch_size 4 --valid_batch_size 4 --test_batch_size 4 --grad_accum 2 --seed $seed --num_sample $num_sample &
        # CUDA_VISIBLE_DEVICES=0 python main_generation.py --lang $lang --model_type ${model_choices_w_checkpoint[$i]} --model_checkpoint ${model_checkpoints[$i]} --device $device --n_epochs $n_epochs --lr $lr --train_batch_size $train_batch_size --valid_batch_size $valid_batch_size --test_batch_size $test_batch_size --seed $seed --num_sample $num_sample
    done

    for ((i = 0; i < ${#model_choices_w_checkpoint[@]}; ++i)); do
        wait
    done
    
    rm -r save/nusa_kalimat/mt/$lang/*/checkpoint*
done

# mt w/ models checkpoint for baseline-mbart and baseline-mt5
model_checkpoints=(facebook/mbart-large-50)
model_choices_w_checkpoint=(baseline-mbart)
for lang in abs bew bhp btk jav mad mak min mui rej sun; do
    for ((i = 0; i < ${#model_choices_w_checkpoint[@]}; ++i)); do
        echo "Executing MT task on lang ${lang} and model choice ${model_choices[$i]}"
        CUDA_VISIBLE_DEVICES=0 python main_generation.py --lang $lang --model_type ${model_choices_w_checkpoint[$i]} --model_checkpoint ${model_checkpoints[$i]} --device $device --n_epochs $n_epochs --lr 2e-5 --gamma 0.98 --train_batch_size 4 --valid_batch_size 4 --test_batch_size 4 --grad_accum 2 --seed $seed --num_sample $num_sample --force &
        # CUDA_VISIBLE_DEVICES=0 python main_generation.py --lang $lang --model_type ${model_choices_w_checkpoint[$i]} --model_checkpoint ${model_checkpoints[$i]} --device $device --n_epochs $n_epochs --lr $lr --train_batch_size $train_batch_size --valid_batch_size $valid_batch_size --test_batch_size $test_batch_size --seed $seed --num_sample $num_sample
    done

    for ((i = 0; i < ${#model_choices_w_checkpoint[@]}; ++i)); do
        wait
    done
    
    rm -r save/nusa_kalimat/mt/$lang/*/checkpoint*
done

# mt with classical methods (for copy, word-substitution, and pbsmt)
model_choices=(copy word-substitution) # add pbsmt if already implemented
for lang in abs bew bhp btk jav mad mak min mui rej sun; do
    for ((i = 0; i < ${#model_choices[@]}; ++i)); do
        echo "Executing MT task on lang ${lang} and model choice ${model_choices[$i]}"
        python main_generation.py --lang $lang --model_type ${model_choices[$i]} --seed $seed --num_sample $num_sample --force
    done
    wait
    rm -r save/nusa_kalimat/mt/$lang/*/checkpoint*
done