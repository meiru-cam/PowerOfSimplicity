# Data Directory

This directory should contain the MIRAI dataset. The data files are not included in the repository due to their size and must be downloaded separately.

## Download Instructions

1. Download the MIRAI dataset from: [Google Drive Link](https://drive.google.com/file/d/1xmSEHZ_wqtBu1AwLpJ8wCDYmT-jRpfrN/view?usp=sharing)

2. Extract the contents to this directory (`data/MIRAI/`)

## Expected Structure

After downloading and extracting, your directory structure should look like:

```
data/
└── MIRAI/
    ├── data_kg.csv              # Event knowledge graph data (~500MB-1GB)
    ├── data_news.csv            # News articles data (~500MB-1GB)
    ├── info/                    # Metadata and mapping files
    │   ├── dict_code2relation.json
    │   └── [other metadata files]
    ├── test/                    # Full test set
    │   └── relation_query.csv   # 705 forecasting queries
    └── test_subset/             # Balanced subset
        └── relation_query.csv   # 100 forecasting queries
```

## File Descriptions

### data_kg.csv
Historical event data from GDELT (2023), containing structured information about international relations and events.

**Columns include**:
- Date information
- Actor 1 (head entity) - country/organization
- Actor 2 (tail entity) - country/organization  
- CAMEO code - standardized event type
- Source URLs
- Additional metadata

### data_news.csv
News article titles and metadata associated with events.

**Columns include**:
- Article URLs
- Titles
- Publication dates
- Associated actors/countries
- Additional metadata

### test/relation_query.csv
Full test set with 705 forecasting queries from November 2023.

**Columns include**:
- QueryId
- DateStr - forecast target date
- DateNLP - natural language date
- Actor1CountryCode, Actor2CountryCode
- Actor1CountryName, Actor2CountryName
- AnswerDict - ground truth relations (JSON format)

### test_subset/relation_query.csv
Balanced subset with 100 queries, sampled to represent diverse:
- Country pairs
- Relation types
- Temporal patterns

## Data Size

- **Total size**: ~1-2 GB (compressed)
- **Extracted size**: ~2-4 GB
- **Disk space needed**: 5-10 GB (including experiment outputs)

## Data Source

This data is derived from:
1. **GDELT Project**: Global Database of Events, Language, and Tone - https://www.gdeltproject.org
2. **MIRAI Processing**: Cleaned, filtered, and structured by MIRAI authors

## Citation

If you use this data, please cite:

1. **MIRAI Framework**:
```bibtex
@misc{ye2024miraievaluatingllmagents,
      title={MIRAI: Evaluating LLM Agents for Event Forecasting}, 
      author={Chenchen Ye and Ziniu Hu and Yihe Deng and Zijie Huang and Mingyu Derek Ma and Yanqiao Zhu and Wei Wang},
      year={2024},
      eprint={2407.01231},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2407.01231}
}
```

2. **GDELT Project**: Include citation and link to https://www.gdeltproject.org

## Terms of Use

According to [GDELT Terms of Use](https://www.gdeltproject.org/about.html#termsofuse):

> GDELT is an open platform for research and analysis of global society and thus all datasets released by the GDELT Project are available for unlimited and unrestricted use for any academic, commercial, or governmental use of any kind without fee.

**Requirements**:
- Any use must include citation to GDELT Project
- Any use must include citation to MIRAI framework
- For research purposes only
- Proper attribution in all publications and derivative works

## Troubleshooting

### Missing Data Files
If you get errors about missing data files:
```bash
# Check if files exist
ls -lh data/MIRAI/

# Should show data_kg.csv and data_news.csv
# If not, download from the link above
```

### File Format Issues
The CSV files use tab (`\t`) as separator. Make sure your CSV reader is configured correctly:
```python
import pandas as pd
data_kg = pd.read_csv('data/MIRAI/data_kg.csv', sep='\t', dtype=str)
```

### Permission Issues
If you get permission errors:
```bash
chmod -R u+rw data/MIRAI/
```

## Questions?

For data-related questions:
- Check the main README.md
- Refer to MIRAI documentation: https://github.com/yecchen/MIRAI
- Open an issue in this repository


