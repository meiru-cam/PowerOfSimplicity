#!/bin/bash

# TEST_CONFIGURATIONS.sh
# This script tests that all main experimental configurations are runnable
# It runs each for just 1 query (--debug mode) to verify syntax and setup

echo "======================================"
echo "Testing Main Experimental Configurations"
echo "======================================"
echo ""

# Check if OPENAI_API_KEY is set (only needed for OpenAI models)
if [ -z "$OPENAI_API_KEY" ]; then
    echo "⚠️  Warning: OPENAI_API_KEY not set. OpenAI model tests will fail."
    echo "   For local models (Llama, Mistral), this is OK."
    echo ""
fi

cd agents

# Counter for tests
total_tests=0
passed_tests=0
failed_tests=0

# Function to run test
run_test() {
    local test_name=$1
    local command=$2
    
    total_tests=$((total_tests + 1))
    echo "========================================"
    echo "Test $total_tests: $test_name"
    echo "Command: $command"
    echo "========================================"
    
    # Run with debug mode (only 1-3 queries) and capture output
    if eval "$command --debug 2>&1 | tee /tmp/test_output.log"; then
        echo "✅ PASSED: $test_name"
        passed_tests=$((passed_tests + 1))
    else
        echo "❌ FAILED: $test_name"
        echo "   Check /tmp/test_output.log for details"
        failed_tests=$((failed_tests + 1))
    fi
    echo ""
}

# Test 1: SimplifiedRAG with GPT-4o-mini (most common setup)
run_test "SimplifiedRAG - Relation + Events (OpenAI)" \
    "python simpleRAG.py --dataset test_subset --model_name gpt-4o-mini \
     --plan simplerag_rel_e_uni --event_k 30 --temperature 0.0 --timediff 1 \
     --output_dir ../output/test_config"

# Test 2: SimplifiedRAG with Llama-3.1-8B (local model)
run_test "SimplifiedRAG - Relation + Events (Llama)" \
    "python simpleRAG.py --dataset test_subset --model_name Llama-3.1-8B-Instruct \
     --plan simplerag_rel_e_uni --event_k 30 --temperature 0.0 --timediff 1 \
     --output_dir ../output/test_config"

# Test 3: SimplifiedRAG with Mistral (local model)
run_test "SimplifiedRAG - Relation + Events (Mistral)" \
    "python simpleRAG.py --dataset test_subset --model_name Mistral-7B-Instruct-v0.2 \
     --plan simplerag_rel_e_uni --event_k 30 --temperature 0.0 --timediff 1 \
     --output_dir ../output/test_config"

# Test 4: SimplifiedRAG - All data types
run_test "SimplifiedRAG - All Data Types (OpenAI)" \
    "python simpleRAG.py --dataset test_subset --model_name gpt-4o-mini \
     --plan simplerag_rel_e_uni_art_uni_title --event_k 30 --temperature 0.0 --timediff 1 \
     --output_dir ../output/test_config"

# Test 5: Direct IO baseline (OpenAI)
run_test "Direct IO Baseline (OpenAI)" \
    "python direct_agents.py --dataset test_subset --model_name gpt-4o-mini \
     --plan direct --temperature 0.0 --timediff 1 \
     --output_dir ../output/test_config"

# Test 6: Chain-of-Thought baseline (OpenAI)
run_test "Chain-of-Thought Baseline (OpenAI)" \
    "python direct_agents.py --dataset test_subset --model_name gpt-4o-mini \
     --plan cot --temperature 0.0 --timediff 1 \
     --output_dir ../output/test_config"

# Test 7: Direct IO with local model
run_test "Direct IO Baseline (Llama)" \
    "python direct_agents.py --dataset test_subset --model_name Llama-3.1-8B-Instruct \
     --plan direct --temperature 0.0 --timediff 1 \
     --output_dir ../output/test_config"

# Test 8: ReAct with single function (OpenAI)
run_test "ReAct - Single Function (OpenAI)" \
    "python react_agents.py --dataset test_subset --model_name gpt-4o-mini \
     --action func --api full --timediff 1 --temperature 0.0 --max_steps 5 \
     --output_dir ../output/test_config"

# Test 9: ReAct with code block (OpenAI)
run_test "ReAct - Code Block (OpenAI)" \
    "python react_agents.py --dataset test_subset --model_name gpt-4o-mini \
     --action block --api full --timediff 1 --temperature 0.0 --max_steps 5 \
     --output_dir ../output/test_config"

# Test 10: Different temporal distances
run_test "SimplifiedRAG - 7 day forecast (OpenAI)" \
    "python simpleRAG.py --dataset test_subset --model_name gpt-4o-mini \
     --plan simplerag_rel_e_uni --event_k 30 --temperature 0.0 --timediff 7 \
     --output_dir ../output/test_config"

echo ""
echo "======================================"
echo "Test Summary"
echo "======================================"
echo "Total tests: $total_tests"
echo "Passed: $passed_tests"
echo "Failed: $failed_tests"
echo ""

if [ $failed_tests -eq 0 ]; then
    echo "✅ All tests passed!"
    exit 0
else
    echo "❌ Some tests failed. Check output above."
    exit 1
fi


