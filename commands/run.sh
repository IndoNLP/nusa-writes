# models=(bert-base-multilingual-uncased indobenchmark/indobert-base-p1 indolem/indobert-base-uncased)
models=(indobenchmark/indobert-base-p1)

## -- NUSA_ALINEA - EMOT -- ##
# for num_sample in 1 5 10 15 20 -1; do
for num_sample in -1; do
    # for dset_name in nusa_kalimat nusa_alinea; do
    for dset_name in nusa_alinea; do
        # for lang in btk bew bug jav mad mak min mui rej sun; do
        for lang in jav; do
            for ((i = 0; i < ${#models[@]}; ++i)); do
                CUDA_VISIBLE_DEVICES=0 python main.py --dataset_name $dset_name --task emot --lang $lang --model_checkpoint ${models[$i]} --n_epochs 100 --lr 1e-5 --train_batch_size 32 --eval_batch_size 32 --seed 0 --num_sample $num_sample --force
                # CUDA_VISIBLE_DEVICES=0 python main.py --dataset_name $dset_name --task emot --lang $lang --model_checkpoint ${models[$i]} --n_epochs 100 --lr 1e-5 --train_batch_size 32 --eval_batch_size 32 --seed 0 --num_sample $num_sample &
                # CUDA_VISIBLE_DEVICES=1 python main.py --task emot --lang $lang --model_checkpoint ${models[$i]} --n_epochs 100 --lr 1e-5 --train_batch_size 32 --eval_batch_size 32 --seed 1 --num_sample $num_sample &
                # CUDA_VISIBLE_DEVICES=2 python main.py --task emot --lang $lang --model_checkpoint ${models[$i]} --n_epochs 100 --lr 1e-5 --train_batch_size 32 --eval_batch_size 32 --seed 2 --num_sample $num_sample &
                # CUDA_VISIBLE_DEVICES=3 python main.py --task emot --lang $lang --model_checkpoint ${models[$i]} --n_epochs 100 --lr 1e-5 --train_batch_size 32 --eval_batch_size 32 --seed 3 --num_sample $num_sample &
                # CUDA_VISIBLE_DEVICES=4 python main.py --task emot --lang $lang --model_checkpoint ${models[$i]} --n_epochs 100 --lr 1e-5 --train_batch_size 32 --eval_batch_size 32 --seed 4 --num_sample $num_sample &
            done
            wait
            rm -r save/sentiment/$lang/*/checkpoint*
        done
    done
done
