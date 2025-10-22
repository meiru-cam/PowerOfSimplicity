# SimpleReact Explained

## What is SimpleReact?

**SimpleReact is NOT about using fewer API functions**. It's a **knowledge distillation** approach where weaker models learn from stronger models' reasoning.

## How It Works

### Step 1: Generate Expert Reasoning (Strong Model)
First, run a strong model (e.g., GPT-4o or Llama-70B) with **full ReAct** multi-turn reasoning:

```bash
cd agents
python react_agents.py --dataset test_subset --model_name gpt-4o-2024-05-13 \
    --action func --api full --timediff 1 --temperature 0.0 \
    --output_dir ../output/react
```

This creates reasoning trajectories containing:
- **Thoughts**: The model's reasoning process
- **Actions**: API calls made (e.g., `get_events()`, `get_news_articles()`)
- **Observations**: Results returned from API calls

### Step 2: Use Expert Traces as Context (Weak Model)
Then, use those reasoning trajectories as **context** for a weaker/cheaper model to make predictions:

```bash
python simpleReact.py --dataset test_subset --model_name gpt-4o-mini \
    --timediff 1 --plan simplerag_thought_observation \
    --output_dir ../output/simpleReact \
    --alias fix_no_title_full_relation
```

The weak model sees the expert's reasoning and makes a **single-turn direct prediction** without needing to do expensive multi-turn API calls itself.

## Three SimpleReact Variants

### 1. `simplerag_thought_observation`
Provides: **Expert's thoughts + Gathered information**

**Context given to weak model:**
```
The expert's thought is:
"I need to check recent relations between USA and China..."

The gathered information is:
Event 1: USA made statement about China...
Event 2: China responded with appeal...
```

### 2. `simplerag_thought_action_observation`
Provides: **Thoughts + API calls + Results**

**Context given to weak model:**
```
The expert's thought is:
"I need to check recent relations..."

The function executed is:
get_events(date_range=..., head_entities=["USA"], tail_entities=["CHN"])

The gathered information is:
Event 1: USA made statement...
Event 2: China responded...
```

### 3. `simplerag_observation`
Provides: **Only the gathered information** (no reasoning shown)

**Context given to weak model:**
```
The function executed is:
get_events(...)

The gathered information is:
Event 1: USA made statement...
Event 2: China responded...
```

## Why Use SimpleReact?

### Advantages
1. **Cost-effective**: Weak models (gpt-4o-mini, Llama-8B) are much cheaper than strong ones
2. **Faster inference**: Single-turn vs multi-turn reduces latency
3. **Learning from experts**: Weak models benefit from strong models' reasoning
4. **No API costs for weak model**: It just reads the context, doesn't call APIs

### Trade-offs
1. **Requires strong model run first**: Must generate expert traces
2. **Fixed context**: Weak model can't explore different reasoning paths
3. **Storage**: Need to save all expert reasoning trajectories

## Example: Full Workflow

### Generate Expert Traces (One-time, expensive)
```bash
# Use GPT-4o to generate reasoning for test subset
python react_agents.py --dataset test_subset \
    --model_name gpt-4o-2024-05-13 \
    --action func --api full --timediff 1 --temperature 0.0 \
    --output_dir ../output/react
    
# This costs ~$X for 100 queries in test_subset
```

### Use Traces with Multiple Weak Models (Cheap)
```bash
# Test with gpt-4o-mini (cheap)
python simpleReact.py --dataset test_subset \
    --model_name gpt-4o-mini --timediff 1 \
    --plan simplerag_thought_observation

# Test with Llama-8B (free if you have GPU)
python simpleReact.py --dataset test_subset \
    --model_name Llama-3.1-8B-Instruct --timediff 1 \
    --plan simplerag_thought_observation

# Test with Mistral-7B
python simpleReact.py --dataset test_subset \
    --model_name Mistral-7B-Instruct-v0.2 --timediff 1 \
    --plan simplerag_thought_observation
```

## Comparison to SimplifiedRAG

| Approach | Data Source | Model Interaction | Use Case |
|----------|-------------|-------------------|----------|
| **SimplifiedRAG** | Raw database (events, news) | Single-turn retrieval | Direct access to data |
| **SimpleReact** | Expert reasoning traces | Single-turn with context | Learning from expert reasoning |
| **Full ReAct** | Raw database | Multi-turn iterative | Maximum flexibility |


## Code Location

- **SimpleReact implementation**: `agents/simpleReact.py`
- **Full ReAct implementation**: `agents/react_agents.py`
- **SimplifiedRAG implementation**: `agents/simpleRAG.py`

## Important Notes

1. **Dependency**: SimpleReact requires pre-generated ReAct outputs
2. **Alignment**: The `--timediff` and `--alias` must match between ReAct and SimpleReact runs
3. **Expert model choice**: Better expert reasoning → better SimpleReact performance
4. **Context window**: Ensure weak model can handle the full expert trace length


