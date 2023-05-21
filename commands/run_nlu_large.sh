#!/bin/bash

models=(indobenchmark/indobert-large-p1 xlm-roberta-large)

## -- NUSA_KALIMAT -- ##
# emot
for num_sample in -1; do
    for dset_name in nusa_kalimat; do
        for lang in abs btk bew bhp jav mad mak min mui rej sun; do
            for task in emot; do
                for ((i = 0; i < ${#models[@]}; ++i)); do
                    CUDA_VISIBLE_DEVICES=0 python main.py --dataset_name $dset_name --task $task --lang $lang --model_checkpoint ${models[$i]} --n_epochs 100 --lr 1e-5 --train_batch_size 4 --eval_batch_size 4 --grad_accum 8 --seed 43 --num_sample $num_sample
                    # CUDA_VISIBLE_DEVICES=0 python main.py --dataset_name $dset_name --task emot --lang $lang --model_checkpoint ${models[$i]} --n_epochs 100 --lr 1e-5 --train_batch_size 32 --eval_batch_size 32 --seed 43 --num_sample $num_sample &
                done
                
                # rm -r save/$dset_name/$task/$lang/*/checkpoint*
            done
        done
    done
done

# senti
for num_sample in -1; do
    for dset_name in nusa_kalimat; do
        for lang in abs btk bew bhp jav mad mak min mui rej sun; do
            for task in senti; do
                for ((i = 0; i < ${#models[@]}; ++i)); do
                    CUDA_VISIBLE_DEVICES=0 python main.py --dataset_name $dset_name --task $task --lang $lang --model_checkpoint ${models[$i]} --n_epochs 100 --lr 1e-5 --train_batch_size 4 --eval_batch_size 4 --grad_accum 8 --seed 43 --num_sample $num_sample
                    # CUDA_VISIBLE_DEVICES=0 python main.py --dataset_name $dset_name --task emot --lang $lang --model_checkpoint ${models[$i]} --n_epochs 100 --lr 1e-5 --train_batch_size 32 --eval_batch_size 32 --seed 43 --num_sample $num_sample &
                done          

                # rm -r save/$dset_name/$task/$lang/*/checkpoint*
            done
        done
    done
done

## -- NUSA_ALINEA -- ##
# paragraph
for num_sample in -1; do
    for dset_name in nusa_alinea; do
        for lang in btk bew bug jav mad mak min mui rej sun; do
            for task in paragraph; do
                for ((i = 0; i < ${#models[@]}; ++i)); do
                    CUDA_VISIBLE_DEVICES=0 python main.py --dataset_name $dset_name --task $task --lang $lang --model_checkpoint ${models[$i]} --n_epochs 100 --lr 1e-5 --train_batch_size 2 --eval_batch_size 2 --grad_accum 32 --seed 43 --num_sample $num_sample
                    # CUDA_VISIBLE_DEVICES=0 python main.py --dataset_name $dset_name --task emot --lang $lang --model_checkpoint ${models[$i]} --n_epochs 100 --lr 1e-5 --train_batch_size 32 --eval_batch_size 32 --seed 43 --num_sample $num_sample &
                done

                # rm -r save/$dset_name/$task/$lang/*/checkpoint*
            done
        done
    done
done

# emot
for num_sample in -1; do
    for dset_name in nusa_alinea; do
        for lang in btk bew bug jav mad mak min mui rej sun; do
            for task in emot; do
                for ((i = 0; i < ${#models[@]}; ++i)); do
                    CUDA_VISIBLE_DEVICES=0 python main.py --dataset_name $dset_name --task $task --lang $lang --model_checkpoint ${models[$i]} --n_epochs 100 --lr 1e-5 --train_batch_size 1 --eval_batch_size 1 --grad_accum 32 --seed 43 --num_sample $num_sample
                    # CUDA_VISIBLE_DEVICES=0 python main.py --dataset_name $dset_name --task emot --lang $lang --model_checkpoint ${models[$i]} --n_epochs 100 --lr 1e-5 --train_batch_size 32 --eval_batch_size 32 --seed 43 --num_sample $num_sample &
                done
                
                # rm -r save/$dset_name/$task/$lang/*/checkpoint*
            done
        done
    done
done

# topic
for num_sample in -1; do
    for dset_name in nusa_alinea; do
        for lang in btk bew bug jav mad mak min mui rej sun; do
            for task in topic; do
                for ((i = 0; i < ${#models[@]}; ++i)); do
                    CUDA_VISIBLE_DEVICES=0 python main.py --dataset_name $dset_name --task $task --lang $lang --model_checkpoint ${models[$i]} --n_epochs 100 --lr 1e-5 --train_batch_size 1 --eval_batch_size 1 --grad_accum 32--seed 43 --num_sample $num_sample
                    # CUDA_VISIBLE_DEVICES=0 python main.py --dataset_name $dset_name --task emot --lang $lang --model_checkpoint ${models[$i]} --n_epochs 100 --lr 1e-5 --train_batch_size 32 --eval_batch_size 32 --seed 43 --num_sample $num_sample &
                done
                
                # rm -r save/$dset_name/$task/$lang/*/checkpoint*
            done
        done
    done
done

# author
for num_sample in -1; do
    for dset_name in nusa_alinea; do
        for lang in btk bew jav mad mak min sun; do
            for task in author; do
                for ((i = 0; i < ${#models[@]}; ++i)); do
                    CUDA_VISIBLE_DEVICES=0 python main.py --dataset_name $dset_name --task $task --lang $lang --model_checkpoint ${models[$i]} --n_epochs 100 --lr 1e-5 --train_batch_size 1 --eval_batch_size 1 --grad_accum 32 --seed 43 --num_sample $num_sample --text_column_name paragraph --label_column_name author_id
                    # CUDA_VISIBLE_DEVICES=0 python main.py --dataset_name $dset_name --task emot --lang $lang --model_checkpoint ${models[$i]} --n_epochs 100 --lr 1e-5 --train_batch_size 32 --eval_batch_size 32 --seed 43 --num_sample $num_sample &
                done
               
                # rm -r save/$dset_name/$task/$lang/*/checkpoint*
            done
        done
    done
done