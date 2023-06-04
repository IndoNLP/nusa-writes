#!/bin/bash

# hyperparams to execute script, pls change it appropriately for your needs
num_sample=4
seed=0

echo "Current directory: $PWD"

## -- NUSA_KALIMAT -- ##
# mt with classical methods (for copy, word-substitution, and pbsmt)
model_choices=(copy word-substitution) # add pbsmt if already implemented
for lang in abs bew bhp btk jav mad mak min mui rej sun; do
    for ((i = 0; i < ${#model_choices[@]}; ++i)); do
        echo "Executing MT task on lang ${lang} and classic benchmark ${model_choices[$i]}"
        # Uncomment for running classic benchmark with sampled dataset
        # python main_generation_classic.py --lang $lang --model_type ${model_choices[$i]} --seed $seed --num_sample $num_sample --force
        
        # Uncomment for running classic benchmark with full dataset
        python main_generation_classic.py --lang $lang --model_type ${model_choices[$i]} --seed $seed --force
    done
    wait
done