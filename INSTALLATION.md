# MIRAI Installation Guide

## Quick Start (Recommended)

For most users, we recommend using **OpenAI models (gpt-4o-mini, gpt-4o)** which are faster, more reliable, and easier to set up.

### 1. Prerequisites

- **Python 3.9+** (tested with Python 3.9)
- **OpenAI API Key** - Get one from [OpenAI Platform](https://platform.openai.com/)

### 2. Basic Installation (OpenAI Models Only)

```bash
# Clone the repository
git clone <your-repo-url>
cd MIRAI

# Create conda environment
conda create -n mirai python=3.9
conda activate mirai

# Install minimal requirements
pip install -r requirements.txt

# Set your OpenAI API key
export OPENAI_API_KEY='your-api-key-here'
```

### 3. Verify Installation

```bash
python verify_setup.py
```

### 4. Quick Test

Run a simple test with 3 queries:

```bash
bash scripts/run_baseline_simpleRAG_working.sh
```

---

## Advanced Installation (Open-Source Models)

⚠️ **WARNING**: Open-source model support via vLLM has known compatibility issues. See `VLLM_TROUBLESHOOTING.md` for details.

### Requirements for Open-Source Models

- **GPU**: NVIDIA GPU with CUDA support (at least 16GB VRAM for 7B models)
- **CUDA 12.1+** recommended
- **PyTorch 2.0+** with CUDA support

### Installation Steps

```bash
# Create conda environment
conda create -n mirai python=3.9
conda activate mirai

# Install PyTorch with CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install full requirements (includes vLLM)
pip install -r requirements_full.txt
```

### Testing Open-Source Models

```bash
# Test with Mistral-7B (may have vLLM issues)
bash scripts/run_baseline_open_simpleRAG.sh
```

**Note**: If you encounter vLLM errors, we recommend using OpenAI models instead.

---

## Installation Verification

After installation, check that everything is set up correctly:

```bash
python verify_setup.py
```

This will verify:
- ✅ Required packages installed
- ✅ Data files present
- ✅ OpenAI API key configured (if using OpenAI models)
- ⚠️ Optional packages (vLLM, etc.)

---

## Package Summary

### Core Packages (Always Required)

| Package | Purpose |
|---------|---------|
| `openai` | OpenAI API client |
| `tiktoken` | Token counting for OpenAI models |
| `pandas` | Data loading and manipulation |
| `numpy` | Numerical operations |
| `langchain` | LLM framework |
| `tqdm` | Progress bars |
| `rank-bm25` | BM25 retrieval algorithm |

### Optional Packages

| Package | Purpose | When Needed |
|---------|---------|-------------|
| `vllm` | Local model inference | Open-source models (Llama, Mistral) |
| `torch` | Deep learning framework | Open-source models |
| `transformers` | Model loading | Open-source models |
| `sentence-transformers` | Embeddings | Enhanced retrieval (api_implementation_new.py) |

---

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'XXX'"

**Solution**: Install missing package:
```bash
pip install <package-name>
```

### Issue: vLLM fails to load models

**Solution**: Use OpenAI models instead (recommended):
```bash
# Use gpt-4o-mini in your scripts
--model_name gpt-4o-mini
```

See `VLLM_TROUBLESHOOTING.md` for more details.

### Issue: "KeyError: 'OPENAI_API_KEY'"

**Solution**: Set your API key:
```bash
export OPENAI_API_KEY='your-api-key-here'
```

Or add it to your `~/.bashrc`:
```bash
echo "export OPENAI_API_KEY='your-api-key-here'" >> ~/.bashrc
source ~/.bashrc
```

---

## Next Steps

1. ✅ **Read the README.md** for usage instructions
2. ✅ **Download the data** from [Google Drive](https://drive.google.com/file/d/1xmSEHZ_wqtBu1AwLpJ8wCDYmT-jRpfrN/view)
3. ✅ **Run example scripts** in `scripts/` directory
4. ✅ **Check VLLM_TROUBLESHOOTING.md** if using open-source models

