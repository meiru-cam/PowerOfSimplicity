# All Experimental Settings from Shell Scripts

## Unique Configurations Identified

### SimplifiedRAG Experiments

#### Data Type Combinations:
1. `simplerag_art_uni_title` - Article titles (uni-directional)
2. `simplerag_art_bi_title` - Article titles (bi-directional)
3. `simplerag_e_uni` - Events only (uni-directional)
4. `simplerag_e_bi` - Events only (bi-directional)
5. `simplerag_rel` - Relation distribution only
6. `simplerag_rel_e_uni` - Relation + Events (uni)
7. `simplerag_rel_e_bi` - Relation + Events (bi)
8. `simplerag_rel_e_uni_art_uni_title` - All three types (uni)
9. `simplerag_rel_e_bi_art_bi_title` - All three types (bi)

**Source**: `run_baseline_mini_simpleRAG.sh`, `run_baseline_mini_zs.sh`

### Temporal Distances:
- timediff=1 (1 day before)
- timediff=7 (1 week before)
- timediff=30 (1 month before)
- timediff=90 (3 months before)

**Source**: `run_baseline_mini_zs.sh`

### Models Tested:
1. **OpenAI**:
   - gpt-4o-mini (most common)
   - gpt-4o-2024-05-13 (implied in some scripts)

2. **Llama**:
   - Meta-Llama-3-8B-Instruct
   - Llama-3.1-8B-Instruct
   - Meta-Llama-3-70B-Instruct-GPTQ

3. **Mistral**:
   - Mistral-7B-Instruct-v0.2

**Source**: `run_baseline_mini_zs.sh`

### Baselines:

#### Direct IO:
```bash
--plan direct --action none --api none
```

#### Chain-of-Thought:
```bash
--plan cot --action none --api none
```

**Source**: `run_baseline_mini_zs.sh`, 

### ReAct (MIRAI):

#### Action Types:
- `--action func` (single function call)
- `--action block` (code block) - mentioned in README but not in mini scripts

#### API Scopes:
- `--api full` (all APIs)
- `--api kg` (events only)
- `--api news` (articles only)

**Source**: `run_baseline_mini_react.sh`

### SimpleReact:

#### Variants:
- `simplerag_thought_observation` - Reasoning + information
- `simplerag_thought_action_observation` - Full trajectory
- `simplerag_observation` - Information only

**Source**: `run_baseline_mini_simpleReact.sh`

### Aliases (Experimental Variants) (deprecated, not in use now):
- `no_title_extract_generate`

### Parameters:
- `--rounds 1` (standard) or `--rounds 10` (self-consistency)
- `--temperature 0.0` (deterministic)
- `--event_k 30` (default)
- `--diversity 0.0` (default)
- `--debug` (for testing)

## Summary of All Experimental Settings

### Core Experiments (from run_baseline_mini_*.sh):

```bash
# 1. SimplifiedRAG - All data combinations (5 main ones)
python simpleRAG.py --plan simplerag_art_uni_title ...
python simpleRAG.py --plan simplerag_rel_e_uni ...
python simpleRAG.py --plan simplerag_rel_e_bi ...
python simpleRAG.py --plan simplerag_rel_e_uni_art_uni_title ...
python simpleRAG.py --plan simplerag_rel_e_bi_art_bi_title ...

# 2. Temporal distances (4 settings)
for timediff in 1 7 30 90; do
    python simpleRAG.py --timediff $timediff ...
done

# 3. Multiple models (3 models)
for model in gpt-4o-mini Mistral-7B-Instruct-v0.2 Meta-Llama-3-8B-Instruct; do
    python direct_agents.py --model_name $model --plan direct ...
    python direct_agents.py --model_name $model --plan cot ...
done

# 4. ReAct baseline
python react_agents.py --action func --api full ...

# 5. 70B model baseline
python direct_agents.py --model_name Meta-Llama-3-70B-Instruct-GPTQ --plan direct ...
```

### Total Unique Configurations:
- **SimplifiedRAG**: 9 data combinations × 4 temporal distances = 36 configs
- **Baselines (Direct/CoT)**: 2 methods × 3-4 models × 4 temporal distances = 24-32 configs
- **ReAct**: Various combinations
- **SimpleReact**: 3 variants

**Total**: ~80-100 unique experimental configurations across all scripts

## Master Experiment Script (Recommended)

I can create a comprehensive script that runs ALL the main experiments you care about. This would capture:

1. All SimplifiedRAG variants
2. All temporal distances
3. All baseline comparisons
4. Different models

Would you like me to create this master script?

---

**Correct files to delete**: See `CORRECT_UNUSED_FILES.md` for full list

