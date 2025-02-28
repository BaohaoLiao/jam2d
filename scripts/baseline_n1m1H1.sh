#!/bin/bash
# For original think

export CUDA_VISIBLE_DEVICES="0,1"

MODEL_NAME_OR_PATH=DeepSeek/DeepSeek-R1-Distill-Qwen-7B
OUTPUT_DIR=DeepSeek-R1-Distill-Qwen-7B

SPLIT="test"
NUM_TEST_SAMPLE=-1


PROMPT_TYPE="deepseek-r1"
DATA_NAME="aime24,aime25,aimo2,math500_level5"
TOKENIZERS_PARALLELISM=false \
python3 -u baseline_v1.py \
    --model_name_or_path ${MODEL_NAME_OR_PATH} \
    --data_dir "./external/qwen25_math_evaluation/data" \
    --data_name ${DATA_NAME} \
    --output_dir ${OUTPUT_DIR} \
    --split ${SPLIT} \
    --prompt_type ${PROMPT_TYPE} \
    --max_tokens_per_call 32768 \
    --num_test_sample ${NUM_TEST_SAMPLE} \
    --seed 0 \
    --temperature 0.6 \
    --n_sampling 1 \
    --top_p 0.95 \
    --start 0 \
    --end -1 \
    --save_outputs \
    --overwrite \


DATA_NAME="math500_level1,math500_level2,math500_level3,math500_level4"
python3 -u baseline_v1.py \
    --model_name_or_path ${MODEL_NAME_OR_PATH} \
    --data_dir "./external/qwen25_math_evaluation/data" \
    --data_name ${DATA_NAME} \
    --output_dir ${OUTPUT_DIR} \
    --split ${SPLIT} \
    --prompt_type ${PROMPT_TYPE} \
    --max_tokens_per_call 16384 \
    --num_test_sample ${NUM_TEST_SAMPLE} \
    --seed 0 \
    --temperature 0.6 \
    --n_sampling 1 \
    --top_p 0.95 \
    --start 0 \
    --end -1 \
    --save_outputs \
    --overwrite \


PROMPT_TYPE="deepseek-r1-no-think-choice"
DATA_NAME="gpqa"
TOKENIZERS_PARALLELISM=false \
python3 -u 2dmaj_batch.py \
    --model_name_or_path ${MODEL_NAME_OR_PATH} \
    --data_dir "./external/qwen25_math_evaluation/data" \
    --data_name ${DATA_NAME} \
    --output_dir ${OUTPUT_DIR} \
    --split ${SPLIT} \
    --prompt_type ${PROMPT_TYPE} \
    --max_tokens_per_call 32768 \
    --num_test_sample ${NUM_TEST_SAMPLE} \
    --seed 0 \
    --temperature 0 \
    --n_sampling 1 \
    --top_p 1 \
    --start 0 \
    --end -1 \
    --save_outputs \
    --overwrite \