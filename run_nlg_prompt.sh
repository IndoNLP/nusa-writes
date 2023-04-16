# Ind to XXX
CUDA_VISIBLE_DEVICES=0 python main_nlg_prompt.py bigscience/bloomz-560m id all 0
CUDA_VISIBLE_DEVICES=0 python main_nlg_prompt.py bigscience/bloomz-1b1 id all 0
CUDA_VISIBLE_DEVICES=0 python main_nlg_prompt.py bigscience/bloomz-1b7 id all 0
CUDA_VISIBLE_DEVICES=0 python main_nlg_prompt.py bigscience/bloomz-3b id all 0
CUDA_VISIBLE_DEVICES=0 python main_nlg_prompt.py bigscience/bloomz-7b1 id all 0

CUDA_VISIBLE_DEVICES=0 python main_nlg_prompt.py bigscience/mt0-small id all 0
CUDA_VISIBLE_DEVICES=0 python main_nlg_prompt.py bigscience/mt0-base id all 0
CUDA_VISIBLE_DEVICES=0 python main_nlg_prompt.py bigscience/mt0-large id all 0
CUDA_VISIBLE_DEVICES=0 python main_nlg_prompt.py bigscience/mt0-xl id all 0

# XXX to Ind
CUDA_VISIBLE_DEVICES=0 python main_nlg_prompt.py bigscience/bloomz-560m all id 0
CUDA_VISIBLE_DEVICES=0 python main_nlg_prompt.py bigscience/bloomz-1b1 all id 0
CUDA_VISIBLE_DEVICES=0 python main_nlg_prompt.py bigscience/bloomz-1b7 all id 0
CUDA_VISIBLE_DEVICES=0 python main_nlg_prompt.py bigscience/bloomz-3b all id 0
CUDA_VISIBLE_DEVICES=0 python main_nlg_prompt.py bigscience/bloomz-7b1 all id 0

CUDA_VISIBLE_DEVICES=0 python main_nlg_prompt.py bigscience/mt0-small id all 0
CUDA_VISIBLE_DEVICES=0 python main_nlg_prompt.py bigscience/mt0-base id all 0
CUDA_VISIBLE_DEVICES=0 python main_nlg_prompt.py bigscience/mt0-large id all 0
CUDA_VISIBLE_DEVICES=0 python main_nlg_prompt.py bigscience/mt0-xl id all 0
