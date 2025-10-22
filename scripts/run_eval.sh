#!/bin/bash

# run_eval.sh - Unified evaluation script for all experiments
# Evaluates model outputs and computes precision/recall/F1 scores

echo "╔════════════════════════════════════════════════════════╗"
echo "║         MIRAI Evaluation Script                        ║"
echo "╚════════════════════════════════════════════════════════╝"
echo ""

# =============================================================================
# Example 1: Direct I/O Baseline (Multiple Models)
# =============================================================================
echo "Running: Direct I/O Baseline Evaluation"

time_diff=7
plan=direct

for model_name in "gpt-4o-mini" "Llama-3.1-8B-Instruct" "Mistral-7B-Instruct-v0.2"
do
    echo "  → Evaluating $model_name..."
    exact_path=output/${plan}/test/${model_name}/timediff${time_diff}-maxsteps0-${plan}-none-none-temp0.0
    
    python agent_evaluation/eval.py \
        --dataset test \
        --model_name $model_name \
        --rounds 1 \
        --temperature 0.0 \
        --timediff $time_diff \
        --plan $plan \
        --action none \
        --api none \
        --exact_output_dir $exact_path \
        --exact_output_eval_dir output_eval/direct/zs_test_${time_diff}_${model_name}
done

echo "✓ Complete! Results in: output_eval/direct/"
echo ""

# =============================================================================
# ADDITIONAL EVALUATION EXAMPLES
# Uncomment and customize the sections below for different experiments
# =============================================================================

# -----------------------------------------------------------------------------
# Example 2: Chain-of-Thought Baseline
# -----------------------------------------------------------------------------
# echo "Running: Chain-of-Thought Baseline Evaluation"
# 
# model_name="gpt-4o-mini"
# time_diff=7
# plan=cot
# 
# exact_path=output/${plan}/test/${model_name}/timediff${time_diff}-maxsteps0-${plan}-none-none-temp0.0
# 
# python agent_evaluation/eval.py \
#     --dataset test \
#     --model_name $model_name \
#     --rounds 1 \
#     --temperature 0.0 \
#     --timediff $time_diff \
#     --plan $plan \
#     --action none \
#     --api none \
#     --exact_output_dir $exact_path \
#     --exact_output_eval_dir output_eval/cot/${plan}_test_${time_diff}
# 
# echo "✓ Complete!"
# echo ""

# -----------------------------------------------------------------------------
# Example 3: SimplifiedRAG Evaluation
# -----------------------------------------------------------------------------
# echo "Running: SimplifiedRAG Evaluation"
# 
# model_name="gpt-4o-mini"
# time_diff=1
# plan="simplerag_rel_e_uni"
# event_k=30
# 
# # Note: Replace [your_run_dir] with actual output directory name
# exact_path=output/simplerag/test_subset/${model_name}/timediff${time_diff}-${plan}-eventK${event_k}-div0.0-temp0.0
# 
# python agent_evaluation/eval.py \
#     --dataset test_subset \
#     --model_name $model_name \
#     --rounds 1 \
#     --temperature 0.0 \
#     --timediff $time_diff \
#     --plan $plan \
#     --event_k $event_k \
#     --event_diversity 0.0 \
#     --exact_output_dir $exact_path \
#     --exact_output_eval_dir output_eval/simplerag
# 
# echo "✓ Complete!"
# echo ""

# =============================================================================
# Usage Notes:
# 
# 1. For exact output path (recommended):
#    Use --exact_output_dir with the full path to your experiment output
# 
# 2. For auto-detection:
#    Use --output_dir and the script will find the matching directory
# 
# 3. Common parameters:
#    --dataset: test or test_subset
#    --model_name: Model used in experiment
#    --timediff: Days before forecast date (1, 7, 30, 90)
#    --plan: Experiment type (direct, cot, react, simplerag_*, etc.)
# 
# 4. Output structure:
#    Results will be saved in --exact_output_eval_dir directory
#    Look for evaluation metrics in the generated JSON/CSV files
# =============================================================================

echo "═══════════════════════════════════════════════════════"
echo "All evaluations complete!"
echo "═══════════════════════════════════════════════════════"

