# Quick Reference Card

## 🚀 One-Page Guide to Your Repository

### What Was Done
✅ Created comprehensive README.md with all experiments documented  
✅ Created file organization guide (FILES_TO_ORGANIZE.md)  
✅ Created .gitignore for clean Git repository  
✅ Created data setup guide (data/DATA_README.md)  
✅ Documented all SimplifiedRAG configurations (9 variants)  
✅ Documented all baselines and comparison methods  
✅ Added proper MIRAI acknowledgements  

### Before You Publish - ACTION REQUIRED

#### 🔴 Critical (Must Do)
1. **Update README.md placeholders:**
   - Line ~3: Add paper arXiv link
   - Line ~289: Add your author names  
   - Line ~290: Add conference/journal
   - Line ~660: Add contact info

2. **Remove API keys from shell scripts:**
   ```bash
   grep -r "sk-proj" agents/*.sh  # Find them
   # Replace with: export OPENAI_API_KEY="your_key_here"
   ```

#### 🟡 Recommended
3. **Organize files** (saves 1-5 GB):
   ```bash
   mkdir -p unused
   mv dataset_construction/ examples/ archive_results/ tmp/ obelics/ unused/
   mv *.ipynb analysis_results.xlsx req.txt unused/ 2>/dev/null || true
   ```

4. **Test one experiment:**
   ```bash
   cd agents
   python simpleRAG.py --dataset test_subset --model_name gpt-4o-mini \
       --plan simplerag_rel_e_uni --temperature 0.0 --timediff 1
   ```

### Main Experiments (Copy-Paste Ready)

```bash
cd /home/mz468/code/MIRAI/agents
export OPENAI_API_KEY="your_key_here"

# SimplifiedRAG (your contribution)
python simpleRAG.py --dataset test --model_name gpt-4o-mini \
    --plan simplerag_rel_e_uni --event_k 30 --temperature 0.0 --timediff 1

# Direct baseline
python direct_agents.py --dataset test --model_name gpt-4o-mini \
    --plan direct --temperature 0.0 --timediff 1

# CoT baseline
python direct_agents.py --dataset test --model_name gpt-4o-mini \
    --plan cot --temperature 0.0 --timediff 1

# ReAct (MIRAI)
python react_agents.py --dataset test --model_name gpt-4o-mini \
    --action func --api full --timediff 1 --temperature 0.0
```

### SimplifiedRAG Configurations

| Config | Data Used | Command |
|--------|-----------|---------|
| Rel only | Statistics | `--plan simplerag_rel` |
| Events (uni) | Events A→B | `--plan simplerag_e_uni` |
| Events (bi) | Events A↔B | `--plan simplerag_e_bi` |
| All (uni) | Rel + Events + News | `--plan simplerag_rel_e_uni_art_uni_title` |
| All (bi) | Rel + Events + News | `--plan simplerag_rel_e_bi_art_bi_title` |

### What is SimpleReact?

**SimpleReact is NOT about using fewer API functions!**

It's a **knowledge distillation** approach:
1. Run strong model (GPT-4o/70B) with full ReAct → generates reasoning traces
2. Use those traces as context for weak models (gpt-4o-mini/8B) → cheap predictions

See `SIMPLEREACT_EXPLAINED.md` for full details.

### File Structure

```
MIRAI/
├── README.md                    ← Read this first! (comprehensive)
├── QUICK_REFERENCE.md          ← This file (quick start)
├── FILES_TO_ORGANIZE.md        ← What to clean up
├── SETUP_SUMMARY.md            ← Setup checklist
├── IMPLEMENTATION_COMPLETE.md  ← Full status report
├── agents/                     ← All experiments here
│   ├── simpleRAG.py           ← Main contribution ⭐
│   └── run_*.sh               ← Quick-start scripts
├── data/
│   └── DATA_README.md         ← How to download data
└── .gitignore                 ← Git configuration
```

### Common Parameters

| Flag | Values | Description |
|------|--------|-------------|
| `--dataset` | `test`, `test_subset` | 705 or 100 queries |
| `--model_name` | `gpt-4o-mini`, `gpt-4o-2024-05-13` | Which LLM |
| `--timediff` | `1`, `7`, `30`, `90` | Days before target |
| `--plan` | See configs above | Experiment type |
| `--temperature` | `0.0`-`1.0` | Sampling temp |

### Data Setup

1. Download: https://drive.google.com/file/d/1xmSEHZ_wqtBu1AwLpJ8wCDYmT-jRpfrN/view
2. Extract to: `data/MIRAI/`
3. Verify:
   ```bash
   ls data/MIRAI/
   # Should show: data_kg.csv, data_news.csv, test/, test_subset/
   ```

### Troubleshooting

**GPU OOM**: Use `gpt-4o-mini` or smaller models  
**API Error**: Check `export OPENAI_API_KEY="..."`  
**Missing Data**: Download from link above  
**Import Error**: `pip install -r requirements.txt`  

### Next Steps

1. [ ] Read README.md (comprehensive guide)
2. [ ] Update placeholders (authors, links)
3. [ ] Remove API keys from scripts
4. [ ] Test experiment
5. [ ] Organize files (optional)
6. [ ] Publish! 🎉

### Getting Help

- **Full docs**: See README.md (main documentation)
- **Data issues**: See data/DATA_README.md
- **Setup help**: See SETUP_SUMMARY.md
- **What's done**: See IMPLEMENTATION_COMPLETE.md
- **MIRAI docs**: https://github.com/yecchen/MIRAI

---

**Quick Start**: `cd agents && bash run_baseline_mini_simpleRAG.sh`  
**Status**: ✅ Ready for review and publication

