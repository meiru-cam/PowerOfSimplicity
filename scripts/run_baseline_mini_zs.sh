#!/bin/bash

# run_baseline_zs.sh - Zero-shot and Chain-of-Thought baselines
# Set your OpenAI API key before running:
# export OPENAI_API_KEY="your_openai_api_key_here"

# article_k set as default -> 15
model_name=gpt-4o-mini

# Run React Full Set
event_k=30
time_diff=7


# for time_diff in 1 7 30 90
for time_diff in 7
    do
    for model_name in "gpt-4o-mini" "Mistral-7B-Instruct-v0.2" "Llama-3.1-8B-Instruct"
        do 
        python agents/direct_agents.py --dataset test_subset --model_name $model_name --rounds 1 --temperature=0.0 --timediff=$time_diff \
                        --plan cot --action none --api none  --output_dir output/cot --debug
        done
    done

# for time_diff in 1 7 30 90
for time_diff in 7
    do
    for model_name in "gpt-4o-mini" "Mistral-7B-Instruct-v0.2" "Llama-3.1-8B-Instruct"
        do 
        python agents/direct_agents.py --dataset test_subset --model_name $model_name --rounds 1 --temperature=0.0 --timediff=$time_diff \
                        --plan direct --action none --api none  --output_dir output/direct --debug
        done
    done

