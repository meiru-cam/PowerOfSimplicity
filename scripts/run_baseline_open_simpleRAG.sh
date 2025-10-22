#!/bin/bash

# run_baseline_open_simpleRAG.sh - SimpleRAG experiments with open-source models
# Set your OpenAI API key before running (for tokenization):
# export OPENAI_API_KEY="your_openai_api_key_here"

# article_k set as default -> 15
event_k=30
time_diff=7

for model_name in "Llama-3.1-8B-Instruct" "Mistral-7B-Instruct-v0.2"
    do
    # for plan in 'art_uni_title' 'rel_e_uni' 'rel_e_bi' 'rel_e_uni_art_uni_title' 'rel_e_bi_art_bi_title'
    for plan in 'rel_e_uni'
        do
        python agents/simpleRAG.py --dataset test --model_name $model_name --rounds 1 --temperature=0.0 --timediff=$time_diff \
                    --plan simplerag_$plan --output_dir output/simplerag \
                    --diversity=0.0 --event_k=$event_k --alias=no_title_extract_generate --debug
        done
    done


