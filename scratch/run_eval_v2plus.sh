#!/bin/bash

export PYTHONPATH="./"
gpu=$1  
seed=42
split="test"
lm_model="/checkpoint/arthur/13873492"
compose_mode="v2plus"

output_path="webshop_logs/${split}_${compose_mode}_editdistance_buylast/"
eval_path="scratch"
mkdir -p "${output_path}"
echo "---> ${output_path}"

cp "${eval_path}/run_eval_v2plus.sh" "${output_path}/"
cp "${eval_path}/eval.py" "${output_path}/"
cp "${eval_path}/data_convert.py" "${output_path}/"
cp "${eval_path}/utils.py" "${output_path}/"

TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES="${gpu}" python "${eval_path}/eval.py" \
    --lm_path "${lm_model}" \
    --beams 5 \
    --state_format "text_rich" \
    --step_limit 100 \
    --max_input_len 2048 \
    --max_output_len 128 \
    --max_num_runs 500 \
    --model_parallelism_size 1 \
    --human_goals \
    --buy_last \
    --set "${split}" \
    --seed "${seed}" \
    --compose_mode "${compose_mode}" \
    --sbert \
    --output_path "${output_path}"