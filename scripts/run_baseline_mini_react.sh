#!/bin/bash

# run_baseline_react.sh
# Set your OpenAI API key before running:
# export OPENAI_API_KEY="your_openai_api_key_here"

# article_k set as default -> 15
model_name=gpt-4o-mini

# Run React Full Set
event_k=30
time_diff=1

python agents/react_agents.py --dataset test_subset --model_name $model_name --rounds 1 --temperature=0.0 --timediff=$time_diff \
                --plan react --action func --api full  --output_dir output/react_full_exp \
                --api_dir "APIs/api_description_full.py"  --debug

