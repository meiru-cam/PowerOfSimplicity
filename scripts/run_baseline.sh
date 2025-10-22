#!/bin/bash

# run_baseline.sh
# Set your OpenAI API key before running:
# export OPENAI_API_KEY="your_openai_api_key_here"

event_k=30
article_k=15
time_diff=1


model_name=gpt-4o-mini
# model_name=DeepSeek-R1-Distill-Llama-8B
# model_name=Mistral-7B-Instruct-v0.2
# model_name=Llama-3.1-8B-Instruct

python agents/react_agents.py --dataset test_subset --model_name $model_name --rounds 1 --temperature=0.0 --timediff=$time_diff \
                    --plan react --action func --api full  --output_dir output/react --alias=no_title_full_relation --debug

python agents/simpleRAG.py --dataset test_subset --model_name $model_name --rounds 1 --temperature=0.0 --timediff=$time_diff \
                    --plan simplerag_e_uni --output_dir output/simplerag_noguide \
                    --diversity=0.0 --event_k=$event_k --alias=no_title_extract_generate --debug

