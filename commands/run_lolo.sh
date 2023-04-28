export CUDA_VISIBLE_DEVICES=0

large_models=(xlm-roberta-large)

# Sentiment
for num_sample in -1; do
    for dset_name in nusa_kalimat; do
        for lang in abs btk bew bhp jav mad mak min mui rej sun; do
            for task in senti; do
                for ((i = 0; i < ${#large_models[@]}; ++i)); do
                    python main_lolo.py --dataset_name $dset_name --task $task --lang $lang --model_checkpoint ${large_models[$i]} --n_epochs 100 --lr 1e-5 --train_batch_size 4 --eval_batch_size 4 --grad_accum 8 --seed 43 --num_sample $num_sample --device cuda &
                    # CUDA_VISIBLE_DEVICES=1 python main_lolo.py --task sentiment --lang $lang --model_checkpoint ${large_models[$i]} --n_epochs 100 --lr 1e-5 --train_batch_size 4 --eval_batch_size 4 --grad_accum 8 --seed 1 --num_sample $num_sample --device cuda
                
                    wait
                    rm -r save/$dset_name/$task/lolo_$lang/*/checkpoint*
                done
            done
        done
    done
done
