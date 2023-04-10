# hyperparams to execute script, pls change it appropriately for ur needs
num_sample=4
train_batch_size=4
test_batch_size=4
seed=0
n_epochs=5
lr=10e-5

## -- NUSA_KALIMAT -- ##
# mt w/o models checkpoint (for indo-bart and indo-gpt)
model_choices = (indo-bart indo-gpt2)
for lang in abs bew bhp btk jav mad mak min mui rej sun; do
    for ((i = 0; i < ${model_choices[@]}; ++i)); do
        python main_generation.py --lang $lang --model_type ${model_choices[$i]} --n_epochs $n_epochs --lr $lr --train_batch_size $train_batch_size --eval_batch_size $eval_batch_size --seed $seed --num_sample $num_sample --force
        # CUDA_VISIBLE_DEVICES=0 python main_generation.py --lang $lang --model_type ${model_choices[$i]} --n_epochs $n_epochs --lr $lr --train_batch_size $train_batch_size --eval_batch_size $eval_batch_size --seed $seed --num_sample $num_sample &
    done
    wait
    rm -r save/nusa_kalimat/mt/$lang/*/checkpoint*
done


# mt w/ models checkpoint (for baseline-mbart and baseline-mt5)
model_checkpoints=(facebook/mbart-large-50-one-to-many-mmt google/mt5-base)
model_choices_w_checkpoint=(baseline-mbart baseline-mt5)
for lang in abs bew bhp btk jav mad mak min mui rej sun; do
    for ((i = 0; i < ${model_choices_w_checkpoint[@]}; ++i)); do
        python main_generation.py --lang $lang --model_type ${model_choices_w_checkpoint[$i]} --model_checkpoint ${model_checkpoints[$i]} --n_epochs $n_epochs --lr $lr --train_batch_size $train_batch_size --eval_batch_size $eval_batch_size --seed $seed --num_sample $num_sample --force
        # CUDA_VISIBLE_DEVICES=0 python main_generation.py --lang $lang --model_type ${model_choices_w_checkpoint[$i]} --model_checkpoint ${model_checkpoints[$i]} --n_epochs $n_epochs --lr $lr --train_batch_size $train_batch_size --eval_batch_size $eval_batch_size --seed $seed --num_sample $num_sample --force
    done
    wait
    rm -r save/nusa_kalimat/mt/$lang/*/checkpoint*
done
