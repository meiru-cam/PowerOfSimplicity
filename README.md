# The Power of Simplicity in LLM-Based Event Forecasting

[![REALM 2025](https://img.shields.io/badge/REALM-2025-blue)](https://aclanthology.org/2025.realm-1.32/)
[![ACL Anthology](https://img.shields.io/badge/ACL-Anthology-red)](https://aclanthology.org/2025.realm-1.32/)
[![License](https://img.shields.io/badge/License-Research-blue.svg)](LICENSE)

**Authors:** Meiru Zhang, Auss Abbood, Zaiqiao Meng, Nigel Collier

> **Note**: This repository builds upon the [MIRAI framework](https://github.com/yecchen/MIRAI) for event forecasting.

## Overview

This repository contains code for our paper **"The Power of Simplicity in LLM-Based Event Forecasting"**. We demonstrate that simplified retrieval-augmented generation (RAG) approaches can achieve comparable or superior forecasting accuracy compared to complex agentic methods, while significantly reducing inference costs.



### Key Contributions

1. **Simplified RAG Outperforms Complex Agents** - Single-turn RAG achieves comparable accuracy at reduced costs vs. multi-turn ReAct agents
2. **Structured Data Matters More Than Semantics** - Event data enhances performance more than semantic information
3. **Model Size Affects Context Utilization** - Larger models leverage enriched contexts better than smaller models


## Quick Start

### Installation

1. **Create environment:**
```bash
conda create -n myenv python=3.10
conda activate myenv
```

2. **Install dependencies:**
Requirements: langchain, pytorch, transformers, vllm. See env.yml for specific versions.

Verify Installation via `python verify_setup.py`

3. **Set up API keys:**
```bash
# For OpenAI models
export OPENAI_API_KEY="your_openai_api_key"

# For HuggingFace models (Llama, Mistral)
huggingface-cli login --token "your_hf_token"
```

### Data Setup

1. **Download the MIRAI dataset:**
   - Download from: https://drive.google.com/file/d/1xmSEHZ_wqtBu1AwLpJ8wCDYmT-jRpfrN/view

2. **Extract to data directory:**
```bash
unzip MIRAI_data.zip -d data/
# This creates data/MIRAI/ with:
# - data_kg.csv (event knowledge graph)
# - data_news.csv (news articles)
# - test/ and test_subset/ (query datasets)
```

## Running Experiments

> **💡 Note:** Python agent scripts can be run from either the MIRAI root directory or the `agents/` directory. Both work correctly.

### SimplifiedRAG (Our Contribution)

**Single command:**
```bash
cd agents
python simpleRAG.py \
    --dataset test_subset \
    --model_name gpt-4o-mini \
    --plan simplerag_rel_e_uni \
    --event_k 30 \
    --temperature 0.0 \
    --timediff 1 \
    --rounds 1 \
    --output_dir ../output/simplerag
```

**Using convenience scripts:**
```bash
# Run all major SimplifiedRAG variants
bash scripts/run_baseline_mini_simpleRAG.sh
```

**Relation statistics only (no events):**
```bash
cd agents
python simpleRAG.py \
    --dataset test_subset \
    --model_name gpt-4o-mini \
    --plan simplerag_rel \
    --event_k 0 \
    --temperature 0.0 \
    --timediff 1 \
    --rounds 1 \
    --output_dir ../output/simplerag_rel
```

### Baseline Methods

**Direct I/O (Zero-shot):**
```bash
cd agents
python direct_agents.py \
    --dataset test_subset \
    --model_name gpt-4o-mini \
    --plan direct \
    --temperature 0.0 \
    --timediff 1 \
    --output_dir ../output/direct
```

**Chain-of-Thought:**
```bash
cd agents
python direct_agents.py \
    --dataset test_subset \
    --model_name gpt-4o-mini \
    --plan cot \
    --temperature 0.0 \
    --timediff 1 \
    --output_dir ../output/cot
```

**ReAct (MIRAI):**
```bash
cd agents
python react_agents.py \
    --dataset test_subset \
    --model_name gpt-4o-mini \
    --action func \
    --api full \
    --timediff 1 \
    --temperature 0.0 \
    --max_steps 20 \
    --output_dir ../output/react
```

## Evaluation

After running experiments, evaluate the results:

```bash
cd agent_evaluation
python eval.py \
    --dataset test_subset \
    --model_name gpt-4o-mini \
    --rounds 1 \
    --temperature 0.0 \
    --timediff 1 \
    --plan simplerag_rel_e_uni \
    --event_k 30 \
    --event_diversity 0.0 \
    --exact_output_dir ../output/simplerag/test_subset/gpt-4o-mini/[your_run_dir] \
    --exact_output_eval_dir ../output_eval/simplerag
```

The evaluation computes:
- First-level CAMEO code precision/recall/F1
- Second-level CAMEO code precision/recall/F1
- Per-query performance metrics

## Configuration Options

### SimplifiedRAG Plans

| Plan | Data Used | Description | Typical `event_k` |
|------|-----------|-------------|-------------------|
| `simplerag_rel` | Relation statistics | Statistical distribution only | `0` |
| `simplerag_e_uni` | Events (uni-directional) | A→B events only | `30` |
| `simplerag_e_bi` | Events (bi-directional) | A→B and B→A events | `30` |
| `simplerag_rel_e_uni` | Relations + Events (uni) | **Recommended** | `30` |
| `simplerag_rel_e_bi` | Relations + Events (bi) | Relations + bidirectional events | `30` |
| `simplerag_art_uni_title` | Article titles (uni) | News article titles only | Any |
| `simplerag_art_bi_title` | Article titles (bi) | Bidirectional article titles | Any |
| `simplerag_rel_e_uni_art_uni_title` | All data types (uni) | Relations + Events + News titles | `30` |
| `simplerag_rel_e_bi_art_bi_title` | All data types (bi) | All data, bidirectional | `30` |

**Note:** For `simplerag_rel`, use `--event_k 0` to retrieve only relation statistics without events.

### Temporal Distances

| `--timediff` | Description |
|--------------|-------------|
| `1` | 1 day before target |
| `7` | 1 week before target |
| `30` | 1 month before target |
| `90` | 3 months before target |

### Supported Models

**OpenAI:**
- `gpt-4o-mini` 
- `gpt-4o-2024-05-13`

**Open-source:**
- `Llama-3.1-8B-Instruct`
- `Meta-Llama-3-70B-Instruct-GPTQ`
- `Mistral-7B-Instruct-v0.2`

### Datasets

- `test_subset` - 100 queries (for quick testing)
- `test` - 705 queries (full evaluation)

## Output Format

Forecasts are saved as JSON with CAMEO codes:

```json
{
    "04": ["042", "040"],  // Economic cooperation
    "03": ["036"],         // Diplomatic cooperation
    "06": ["061"]          // Military cooperation
}
```

- **Keys:** 2-digit first-level CAMEO codes (parent relations)
- **Values:** 3-digit second-level CAMEO codes (child relations)

**Example outputs** are provided in `output/examples/` for reference:
- `output/examples/cot/` - Chain-of-Thought baseline
- `output/examples/react/` - ReAct agent
- `output/examples/simplerag/` - SimplifiedRAG approach

## Repository Structure

```
MIRAI/
├── agents/                  # Main experiment scripts
│   ├── simpleRAG.py        # Our SimplifiedRAG approach
│   ├── direct_agents.py    # Direct/CoT baselines
│   ├── react_agents.py     # ReAct baseline
│   └── simpleReact.py      # SimpleReact variant
│
├── agent_evaluation/        # Evaluation scripts
│   └── eval.py
│
├── agent_prompts/           # Prompt templates (18 files)
├── APIs/                    # API implementations (6 files)
│
├── scripts/                 # Helper bash scripts (6 files)
│   ├── run_baseline.sh                   # Quick baseline test
│   ├── run_baseline_mini_simpleRAG.sh   # SimplifiedRAG variants
│   ├── run_baseline_mini_react.sh       # ReAct experiments
│   ├── run_baseline_mini_zs.sh          # Zero-shot & CoT
│   ├── run_baseline_open_simpleRAG.sh   # Open-source models
│   └── run_eval.sh                       # Evaluation script
│
├── data/                    # Dataset directory
│   ├── MIRAI/              # MIRAI dataset (download required)
│   └── info/               # Reference data (countries, relations)
│
├── output/                  # Experiment outputs
│   └── examples/           # Example outputs for each method
│
├── README.md               # This file
├── INSTALLATION.md         # Detailed setup instructions
├── QUICK_REFERENCE.md      # Quick start guide
├── ALL_EXPERIMENTAL_SETTINGS.md  # All configuration options
├── requirements.txt        # Python dependencies
└── TEST_CONFIGURATIONS.sh  # Quick test script
```

## Quick Test

To verify everything works:

```bash
# Run test configuration (3 queries)
bash TEST_CONFIGURATIONS.sh

# Or run a single test
cd agents
python simpleRAG.py \
    --dataset test_subset \
    --model_name gpt-4o-mini \
    --plan simplerag_rel_e_uni \
    --event_k 30 \
    --temperature 0.0 \
    --timediff 1 \
    --debug
```

## Common Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--dataset` | Dataset to use | `test_subset` |
| `--model_name` | LLM model | Required |
| `--temperature` | Sampling temperature | `0.0` |
| `--timediff` | Days before target date | `1` |
| `--rounds` | Number of rounds | `1` |
| `--event_k` | Top-k events to retrieve | `30` |
| `--output_dir` | Output directory | Required |
| `--debug` | Run on 3 queries only | Flag |

## Troubleshooting

**GPU Out of Memory:**
- Use smaller models (`gpt-4o-mini`, `Llama-3.1-8B-Instruct`)
- Reduce `--event_k` value

**API Key Error:**
- Verify: `echo $OPENAI_API_KEY`
- Set in current session: `export OPENAI_API_KEY="your_key"`

**Missing Data:**
- Ensure `data/MIRAI/` exists with CSV files
- Check: `ls data/MIRAI/`

**Import Errors:**
- Reinstall: `pip install -r requirements.txt`
- Check Python version: `python --version` (should be 3.9+)

## Citation

If you use this code in your research, please cite:

```bibtex
@inproceedings{zhang2025power,
  title={The Power of Simplicity in LLM-Based Event Forecasting},
  author={Zhang, Meiru and Abbood, Auss and Meng, Zaiqiao and Collier, Nigel},
  booktitle={Proceedings of the 1st Workshop for Research on Agent Language Models (REALM 2025)},
  pages={454--470},
  year={2025}
}
```

## Acknowledgements

This repository builds upon the [MIRAI framework](https://github.com/yecchen/MIRAI):

```bibtex
@misc{ye2024miraievaluatingllmagents,
      title={MIRAI: Evaluating LLM Agents for Event Forecasting}, 
      author={Chenchen Ye and Ziniu Hu and Yihe Deng and Zijie Huang and Mingyu Derek Ma and Yanqiao Zhu and Wei Wang},
      year={2024},
      eprint={2407.01231},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2407.01231}, 
}
```

## License

This project is for research purposes. Please see LICENSE file for details.

## Contact

For questions or issues, please open an issue on GitHub or contact mz468@cam.ac.uk.

---

**See also:**
- `INSTALLATION.md` - Detailed installation instructions
- `QUICK_REFERENCE.md` - Quick start commands
- `ALL_EXPERIMENTAL_SETTINGS.md` - All configuration options
- `SIMPLEREACT_EXPLAINED.md` - SimpleReact methodology
- `output/examples/README.md` - Example output format guide

