#!/bin/bash

# run_baseline_simpleRAG.sh
# Set your OpenAI API key before running:
# export OPENAI_API_KEY="your_openai_api_key_here"

# article_k set as default -> 15
model_name=gpt-4o-mini
event_k=30
time_diff=7

# rel_e_uni_art_uni_title
# rel_e_bi_art_bi_title

for plan in 'art_uni_title' 'rel_e_uni' 'rel_e_bi' 'rel_e_uni_art_uni_title' 'rel_e_bi_art_bi_title'
    do
    python agents/simpleRAG.py --dataset test --model_name $model_name --rounds 1 --temperature=0.0 --timediff=$time_diff \
                --plan simplerag_$plan --output_dir output/simplerag \
                --diversity=0.0 --event_k=$event_k --alias=no_title_extract_generate --debug
    done


