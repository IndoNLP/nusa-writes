###
# Wikipedia
###
CUDA_VISIBLE_DEVICES=1 python run_clm.py \
    --model_name_or_path indobenchmark/indogpt \
    --per_device_train_batch_size 128 \
    --per_device_eval_batch_size 128 \
    --max_steps 5000 \
    --do_train \
    --do_eval \
    --rebalance \
    --logging_steps 10 \
    --evaluation_strategy steps \
    --eval_steps 300 \
    --save_strategy no \
    --learning_rate 1e-4 \
    --preprocessing_num_workers 32 \
    --dataloader_num_workers 32 \
    --overwrite_output_dir \
    --fp16 \
    --seed 42 \
    --dataset_name wikipedia \
    --dataset_lang bug \
    --output_dir ./save/wikipedia/bug

CUDA_VISIBLE_DEVICES=1 python run_clm.py \
    --model_name_or_path indobenchmark/indogpt \
    --per_device_train_batch_size 128 \
    --per_device_eval_batch_size 128 \
    --max_steps 5000 \
    --do_train \
    --do_eval \
    --rebalance \
    --logging_steps 10 \
    --evaluation_strategy steps \
    --eval_steps 300 \
    --save_strategy no \
    --learning_rate 1e-4 \
    --preprocessing_num_workers 32 \
    --dataloader_num_workers 32 \
    --overwrite_output_dir \
    --fp16 \
    --seed 42 \
    --dataset_name wikipedia \
    --dataset_lang mad \
    --output_dir ./save/wikipedia/mad

CUDA_VISIBLE_DEVICES=1 python run_clm.py \
    --model_name_or_path indobenchmark/indogpt \
    --per_device_train_batch_size 128 \
    --per_device_eval_batch_size 128 \
    --max_steps 5000 \
    --do_train \
    --do_eval \
    --rebalance \
    --logging_steps 10 \
    --evaluation_strategy steps \
    --eval_steps 300 \
    --save_strategy no \
    --learning_rate 1e-4 \
    --preprocessing_num_workers 32 \
    --dataloader_num_workers 32 \
    --overwrite_output_dir \
    --fp16 \
    --seed 42 \
    --dataset_name wikipedia \
    --dataset_lang sun \
    --output_dir ./save/wikipedia/sun

CUDA_VISIBLE_DEVICES=1 python run_clm.py \
    --model_name_or_path indobenchmark/indogpt \
    --per_device_train_batch_size 128 \
    --per_device_eval_batch_size 128 \
    --max_steps 5000 \
    --do_train \
    --do_eval \
    --rebalance \
    --logging_steps 10 \
    --evaluation_strategy steps \
    --eval_steps 300 \
    --save_strategy no \
    --learning_rate 1e-4 \
    --preprocessing_num_workers 32 \
    --dataloader_num_workers 32 \
    --overwrite_output_dir \
    --fp16 \
    --seed 42 \
    --dataset_name wikipedia \
    --dataset_lang jav \
    --output_dir ./save/wikipedia/jav

CUDA_VISIBLE_DEVICES=1 python run_clm.py \
    --model_name_or_path indobenchmark/indogpt \
    --per_device_train_batch_size 128 \
    --per_device_eval_batch_size 128 \
    --max_steps 5000 \
    --do_train \
    --do_eval \
    --rebalance \
    --logging_steps 10 \
    --evaluation_strategy steps \
    --eval_steps 300 \
    --save_strategy no \
    --learning_rate 1e-4 \
    --preprocessing_num_workers 32 \
    --dataloader_num_workers 32 \
    --overwrite_output_dir \
    --fp16 \
    --seed 42 \
    --dataset_name wikipedia \
    --dataset_lang min \
    --output_dir ./save/wikipedia/min

###
# Paragraph
###
CUDA_VISIBLE_DEVICES=1 python run_clm.py \
    --model_name_or_path indobenchmark/indogpt \
    --per_device_train_batch_size 128 \
    --per_device_eval_batch_size 128 \
    --max_steps 5000 \
    --do_train \
    --do_eval \
    --rebalance \
    --logging_steps 10 \
    --evaluation_strategy steps \
    --eval_steps 300 \
    --save_strategy no \
    --learning_rate 1e-4 \
    --preprocessing_num_workers 32 \
    --dataloader_num_workers 32 \
    --overwrite_output_dir \
    --fp16 \
    --seed 42 \
    --dataset_name paragraph \
    --dataset_lang bug \
    --output_dir ./save/paragraph/bug

CUDA_VISIBLE_DEVICES=1 python run_clm.py \
    --model_name_or_path indobenchmark/indogpt \
    --per_device_train_batch_size 128 \
    --per_device_eval_batch_size 128 \
    --max_steps 5000 \
    --do_train \
    --do_eval \
    --rebalance \
    --logging_steps 10 \
    --evaluation_strategy steps \
    --eval_steps 300 \
    --save_strategy no \
    --learning_rate 1e-4 \
    --preprocessing_num_workers 32 \
    --dataloader_num_workers 32 \
    --overwrite_output_dir \
    --fp16 \
    --seed 42 \
    --dataset_name paragraph \
    --dataset_lang mad \
    --output_dir ./save/paragraph/mad

CUDA_VISIBLE_DEVICES=1 python run_clm.py \
    --model_name_or_path indobenchmark/indogpt \
    --per_device_train_batch_size 128 \
    --per_device_eval_batch_size 128 \
    --max_steps 5000 \
    --do_train \
    --do_eval \
    --rebalance \
    --logging_steps 10 \
    --evaluation_strategy steps \
    --eval_steps 300 \
    --save_strategy no \
    --learning_rate 1e-4 \
    --preprocessing_num_workers 32 \
    --dataloader_num_workers 32 \
    --overwrite_output_dir \
    --fp16 \
    --seed 42 \
    --dataset_name paragraph \
    --dataset_lang sun \
    --output_dir ./save/paragraph/sun

CUDA_VISIBLE_DEVICES=1 python run_clm.py \
    --model_name_or_path indobenchmark/indogpt \
    --per_device_train_batch_size 128 \
    --per_device_eval_batch_size 128 \
    --max_steps 5000 \
    --do_train \
    --do_eval \
    --rebalance \
    --logging_steps 10 \
    --evaluation_strategy steps \
    --eval_steps 300 \
    --save_strategy no \
    --learning_rate 1e-4 \
    --preprocessing_num_workers 32 \
    --dataloader_num_workers 32 \
    --overwrite_output_dir \
    --fp16 \
    --seed 42 \
    --dataset_name paragraph \
    --dataset_lang jav \
    --output_dir ./save/paragraph/jav

CUDA_VISIBLE_DEVICES=1 python run_clm.py \
    --model_name_or_path indobenchmark/indogpt \
    --per_device_train_batch_size 128 \
    --per_device_eval_batch_size 128 \
    --max_steps 5000 \
    --do_train \
    --do_eval \
    --rebalance \
    --logging_steps 10 \
    --evaluation_strategy steps \
    --eval_steps 300 \
    --save_strategy no \
    --learning_rate 1e-4 \
    --preprocessing_num_workers 32 \
    --dataloader_num_workers 32 \
    --overwrite_output_dir \
    --fp16 \
    --seed 42 \
    --dataset_name paragraph \
    --dataset_lang min \
    --output_dir ./save/paragraph/min

###
# Translation
###
CUDA_VISIBLE_DEVICES=1 python run_clm.py \
    --model_name_or_path indobenchmark/indogpt \
    --per_device_train_batch_size 128 \
    --per_device_eval_batch_size 128 \
    --max_steps 5000 \
    --do_train \
    --do_eval \
    --rebalance \
    --logging_steps 10 \
    --evaluation_strategy steps \
    --eval_steps 300 \
    --save_strategy no \
    --learning_rate 1e-4 \
    --preprocessing_num_workers 32 \
    --dataloader_num_workers 32 \
    --overwrite_output_dir \
    --fp16 \
    --seed 42 \
    --dataset_name translation \
    --dataset_lang bug \
    --output_dir ./save/translation/bug

CUDA_VISIBLE_DEVICES=1 python run_clm.py \
    --model_name_or_path indobenchmark/indogpt \
    --per_device_train_batch_size 128 \
    --per_device_eval_batch_size 128 \
    --max_steps 5000 \
    --do_train \
    --do_eval \
    --rebalance \
    --logging_steps 10 \
    --evaluation_strategy steps \
    --eval_steps 300 \
    --save_strategy no \
    --learning_rate 1e-4 \
    --preprocessing_num_workers 32 \
    --dataloader_num_workers 32 \
    --overwrite_output_dir \
    --fp16 \
    --seed 42 \
    --dataset_name translation \
    --dataset_lang mad \
    --output_dir ./save/translation/mad

CUDA_VISIBLE_DEVICES=1 python run_clm.py \
    --model_name_or_path indobenchmark/indogpt \
    --per_device_train_batch_size 128 \
    --per_device_eval_batch_size 128 \
    --max_steps 5000 \
    --do_train \
    --do_eval \
    --rebalance \
    --logging_steps 10 \
    --evaluation_strategy steps \
    --eval_steps 300 \
    --save_strategy no \
    --learning_rate 1e-4 \
    --preprocessing_num_workers 32 \
    --dataloader_num_workers 32 \
    --overwrite_output_dir \
    --fp16 \
    --seed 42 \
    --dataset_name translation \
    --dataset_lang sun \
    --output_dir ./save/translation/sun

CUDA_VISIBLE_DEVICES=1 python run_clm.py \
    --model_name_or_path indobenchmark/indogpt \
    --per_device_train_batch_size 128 \
    --per_device_eval_batch_size 128 \
    --max_steps 5000 \
    --do_train \
    --do_eval \
    --rebalance \
    --logging_steps 10 \
    --evaluation_strategy steps \
    --eval_steps 300 \
    --save_strategy no \
    --learning_rate 1e-4 \
    --preprocessing_num_workers 32 \
    --dataloader_num_workers 32 \
    --overwrite_output_dir \
    --fp16 \
    --seed 42 \
    --dataset_name translation \
    --dataset_lang jav \
    --output_dir ./save/translation/jav

CUDA_VISIBLE_DEVICES=1 python run_clm.py \
    --model_name_or_path indobenchmark/indogpt \
    --per_device_train_batch_size 128 \
    --per_device_eval_batch_size 128 \
    --max_steps 5000 \
    --do_train \
    --do_eval \
    --rebalance \
    --logging_steps 10 \
    --evaluation_strategy steps \
    --eval_steps 300 \
    --save_strategy no \
    --learning_rate 1e-4 \
    --preprocessing_num_workers 32 \
    --dataloader_num_workers 32 \
    --overwrite_output_dir \
    --fp16 \
    --seed 42 \
    --dataset_name translation \
    --dataset_lang min \
    --output_dir ./save/translation/min

###
# Wikipedia Full
###
CUDA_VISIBLE_DEVICES=1 python run_clm.py \
    --model_name_or_path indobenchmark/indogpt \
    --per_device_train_batch_size 128 \
    --per_device_eval_batch_size 128 \
    --max_steps 5000 \
    --do_train \
    --do_eval \
    --logging_steps 10 \
    --evaluation_strategy steps \
    --eval_steps 300 \
    --save_strategy no \
    --learning_rate 1e-4 \
    --preprocessing_num_workers 32 \
    --dataloader_num_workers 32 \
    --overwrite_output_dir \
    --fp16 \
    --seed 42 \
    --dataset_name wikipedia \
    --dataset_lang bug \
    --output_dir ./save_full/wikipedia/bug

CUDA_VISIBLE_DEVICES=1 python run_clm.py \
    --model_name_or_path indobenchmark/indogpt \
    --per_device_train_batch_size 128 \
    --per_device_eval_batch_size 128 \
    --max_steps 5000 \
    --do_train \
    --do_eval \
    --logging_steps 10 \
    --evaluation_strategy steps \
    --eval_steps 300 \
    --save_strategy no \
    --learning_rate 1e-4 \
    --preprocessing_num_workers 32 \
    --dataloader_num_workers 32 \
    --overwrite_output_dir \
    --fp16 \
    --seed 42 \
    --dataset_name wikipedia \
    --dataset_lang mad \
    --output_dir ./save_full/wikipedia/mad

CUDA_VISIBLE_DEVICES=1 python run_clm.py \
    --model_name_or_path indobenchmark/indogpt \
    --per_device_train_batch_size 128 \
    --per_device_eval_batch_size 128 \
    --max_steps 5000 \
    --do_train \
    --do_eval \
    --logging_steps 10 \
    --evaluation_strategy steps \
    --eval_steps 300 \
    --save_strategy no \
    --learning_rate 1e-4 \
    --preprocessing_num_workers 32 \
    --dataloader_num_workers 32 \
    --overwrite_output_dir \
    --fp16 \
    --seed 42 \
    --dataset_name wikipedia \
    --dataset_lang sun \
    --output_dir ./save_full/wikipedia/sun

CUDA_VISIBLE_DEVICES=1 python run_clm.py \
    --model_name_or_path indobenchmark/indogpt \
    --per_device_train_batch_size 128 \
    --per_device_eval_batch_size 128 \
    --max_steps 5000 \
    --do_train \
    --do_eval \
    --logging_steps 10 \
    --evaluation_strategy steps \
    --eval_steps 300 \
    --save_strategy no \
    --learning_rate 1e-4 \
    --preprocessing_num_workers 32 \
    --dataloader_num_workers 32 \
    --overwrite_output_dir \
    --fp16 \
    --seed 42 \
    --dataset_name wikipedia \
    --dataset_lang jav \
    --output_dir ./save_full/wikipedia/jav

CUDA_VISIBLE_DEVICES=1 python run_clm.py \
    --model_name_or_path indobenchmark/indogpt \
    --per_device_train_batch_size 128 \
    --per_device_eval_batch_size 128 \
    --max_steps 5000 \
    --do_train \
    --do_eval \
    --logging_steps 10 \
    --evaluation_strategy steps \
    --eval_steps 300 \
    --save_strategy no \
    --learning_rate 1e-4 \
    --preprocessing_num_workers 32 \
    --dataloader_num_workers 32 \
    --overwrite_output_dir \
    --fp16 \
    --seed 42 \
    --dataset_name wikipedia \
    --dataset_lang min \
    --output_dir ./save_full/wikipedia/min

###
# Paragraph Full
###
CUDA_VISIBLE_DEVICES=1 python run_clm.py \
    --model_name_or_path indobenchmark/indogpt \
    --per_device_train_batch_size 128 \
    --per_device_eval_batch_size 128 \
    --max_steps 5000 \
    --do_train \
    --do_eval \
    --logging_steps 10 \
    --evaluation_strategy steps \
    --eval_steps 300 \
    --save_strategy no \
    --learning_rate 1e-4 \
    --preprocessing_num_workers 32 \
    --dataloader_num_workers 32 \
    --overwrite_output_dir \
    --fp16 \
    --seed 42 \
    --dataset_name paragraph \
    --dataset_lang bug \
    --output_dir ./save_full/paragraph/bug

CUDA_VISIBLE_DEVICES=1 python run_clm.py \
    --model_name_or_path indobenchmark/indogpt \
    --per_device_train_batch_size 128 \
    --per_device_eval_batch_size 128 \
    --max_steps 5000 \
    --do_train \
    --do_eval \
    --logging_steps 10 \
    --evaluation_strategy steps \
    --eval_steps 300 \
    --save_strategy no \
    --learning_rate 1e-4 \
    --preprocessing_num_workers 32 \
    --dataloader_num_workers 32 \
    --overwrite_output_dir \
    --fp16 \
    --seed 42 \
    --dataset_name paragraph \
    --dataset_lang mad \
    --output_dir ./save_full/paragraph/mad

CUDA_VISIBLE_DEVICES=1 python run_clm.py \
    --model_name_or_path indobenchmark/indogpt \
    --per_device_train_batch_size 128 \
    --per_device_eval_batch_size 128 \
    --max_steps 5000 \
    --do_train \
    --do_eval \
    --logging_steps 10 \
    --evaluation_strategy steps \
    --eval_steps 300 \
    --save_strategy no \
    --learning_rate 1e-4 \
    --preprocessing_num_workers 32 \
    --dataloader_num_workers 32 \
    --overwrite_output_dir \
    --fp16 \
    --seed 42 \
    --dataset_name paragraph \
    --dataset_lang sun \
    --output_dir ./save_full/paragraph/sun

CUDA_VISIBLE_DEVICES=1 python run_clm.py \
    --model_name_or_path indobenchmark/indogpt \
    --per_device_train_batch_size 128 \
    --per_device_eval_batch_size 128 \
    --max_steps 5000 \
    --do_train \
    --do_eval \
    --logging_steps 10 \
    --evaluation_strategy steps \
    --eval_steps 300 \
    --save_strategy no \
    --learning_rate 1e-4 \
    --preprocessing_num_workers 32 \
    --dataloader_num_workers 32 \
    --overwrite_output_dir \
    --fp16 \
    --seed 42 \
    --dataset_name paragraph \
    --dataset_lang jav \
    --output_dir ./save_full/paragraph/jav

CUDA_VISIBLE_DEVICES=1 python run_clm.py \
    --model_name_or_path indobenchmark/indogpt \
    --per_device_train_batch_size 128 \
    --per_device_eval_batch_size 128 \
    --max_steps 5000 \
    --do_train \
    --do_eval \
    --logging_steps 10 \
    --evaluation_strategy steps \
    --eval_steps 300 \
    --save_strategy no \
    --learning_rate 1e-4 \
    --preprocessing_num_workers 32 \
    --dataloader_num_workers 32 \
    --overwrite_output_dir \
    --fp16 \
    --seed 42 \
    --dataset_name paragraph \
    --dataset_lang min \
    --output_dir ./save_full/paragraph/min

###
# Translation Full
###
CUDA_VISIBLE_DEVICES=1 python run_clm.py \
    --model_name_or_path indobenchmark/indogpt \
    --per_device_train_batch_size 128 \
    --per_device_eval_batch_size 128 \
    --max_steps 5000 \
    --do_train \
    --do_eval \
    --logging_steps 10 \
    --evaluation_strategy steps \
    --eval_steps 300 \
    --save_strategy no \
    --learning_rate 1e-4 \
    --preprocessing_num_workers 32 \
    --dataloader_num_workers 32 \
    --overwrite_output_dir \
    --fp16 \
    --seed 42 \
    --dataset_name translation \
    --dataset_lang bug \
    --output_dir ./save_full/translation/bug

CUDA_VISIBLE_DEVICES=1 python run_clm.py \
    --model_name_or_path indobenchmark/indogpt \
    --per_device_train_batch_size 128 \
    --per_device_eval_batch_size 128 \
    --max_steps 5000 \
    --do_train \
    --do_eval \
    --logging_steps 10 \
    --evaluation_strategy steps \
    --eval_steps 300 \
    --save_strategy no \
    --learning_rate 1e-4 \
    --preprocessing_num_workers 32 \
    --dataloader_num_workers 32 \
    --overwrite_output_dir \
    --fp16 \
    --seed 42 \
    --dataset_name translation \
    --dataset_lang mad \
    --output_dir ./save_full/translation/mad

CUDA_VISIBLE_DEVICES=1 python run_clm.py \
    --model_name_or_path indobenchmark/indogpt \
    --per_device_train_batch_size 128 \
    --per_device_eval_batch_size 128 \
    --max_steps 5000 \
    --do_train \
    --do_eval \
    --logging_steps 10 \
    --evaluation_strategy steps \
    --eval_steps 300 \
    --save_strategy no \
    --learning_rate 1e-4 \
    --preprocessing_num_workers 32 \
    --dataloader_num_workers 32 \
    --overwrite_output_dir \
    --fp16 \
    --seed 42 \
    --dataset_name translation \
    --dataset_lang sun \
    --output_dir ./save_full/translation/sun

CUDA_VISIBLE_DEVICES=1 python run_clm.py \
    --model_name_or_path indobenchmark/indogpt \
    --per_device_train_batch_size 128 \
    --per_device_eval_batch_size 128 \
    --max_steps 5000 \
    --do_train \
    --do_eval \
    --logging_steps 10 \
    --evaluation_strategy steps \
    --eval_steps 300 \
    --save_strategy no \
    --learning_rate 1e-4 \
    --preprocessing_num_workers 32 \
    --dataloader_num_workers 32 \
    --overwrite_output_dir \
    --fp16 \
    --seed 42 \
    --dataset_name translation \
    --dataset_lang jav \
    --output_dir ./save_full/translation/jav

CUDA_VISIBLE_DEVICES=1 python run_clm.py \
    --model_name_or_path indobenchmark/indogpt \
    --per_device_train_batch_size 128 \
    --per_device_eval_batch_size 128 \
    --max_steps 5000 \
    --do_train \
    --do_eval \
    --logging_steps 10 \
    --evaluation_strategy steps \
    --eval_steps 300 \
    --save_strategy no \
    --learning_rate 1e-4 \
    --preprocessing_num_workers 32 \
    --dataloader_num_workers 32 \
    --overwrite_output_dir \
    --fp16 \
    --seed 42 \
    --dataset_name translation \
    --dataset_lang min \
    --output_dir ./save_full/translation/min
