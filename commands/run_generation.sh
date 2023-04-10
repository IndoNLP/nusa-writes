## -- NUSA_KALIMAT -- ##
# mt w/o models checkpoint (for indo-bart and indo-gpt)
model_choices = (indo-bart indo-gpt2)
for num_sample in -1; do
  for lang in abs bew bhp btk jav mad mak min mui rej sun; do
          for ((i = 0; i < ${model_choices[@]}; ++i)); do
              python main_generation.py --lang $lang --model_type ${model_choices[$i]} --n_epochs 100 --lr 1e-5 --train_batch_size 32 --eval_batch_size 32 --seed 0 --num_sample $num_sample --force
              # CUDA_VISIBLE_DEVICES=0 python main.py --lang $lang --model_checkpoint ${models[$i]} --n_epochs 100 --lr 1e-5 --train_batch_size 32 --eval_batch_size 32 --seed 0 --num_sample $num_sample &
          done
          wait
          rm -r save/nusa_kalimat/mt/$lang/*/checkpoint*
      done
  done
done


# mt w/ models checkpoint (for baseline-mbart and baseline-mt5)
model_checkpoints=(facebook/mbart-large-50-one-to-many-mmt google/mt5-base)
model_choices_w_checkpoint=(baseline-mbart baseline-mt5)
for num_sample in -1; do
  for lang in abs bew bhp btk jav mad mak min mui rej sun; do
          for ((i = 0; i < ${model_choices_w_checkpoint[@]}; ++i)); do
              python main_generation.py --lang $lang --model_type ${model_choices_w_checkpoint[$i]} --model_checkpoint ${model_checkpoints[$i]} --n_epochs 100 --lr 1e-5 --train_batch_size 32 --eval_batch_size 32 --seed 0 --num_sample $num_sample --force
              # CUDA_VISIBLE_DEVICES=0 python main.py --lang $lang --model_checkpoint ${models[$i]} --n_epochs 100 --lr 1e-5 --train_batch_size 32 --eval_batch_size 32 --seed 0 --num_sample $num_sample &
          done
          wait
          rm -r save/nusa_kalimat/mt/$lang/*/checkpoint*
      done
  done
done