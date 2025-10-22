#!/usr/bin/env python3
"""
MIRAI Setup Verification Script
Checks that all required packages and data are properly installed
"""

import sys
import os
from pathlib import Path

def check_package(package_name, optional=False):
    """Check if a package is installed"""
    try:
        __import__(package_name)
        print(f"✅ {package_name:<30} {'(optional)' if optional else 'INSTALLED'}")
        return True
    except ImportError:
        if optional:
            print(f"⚠️  {package_name:<30} (optional, not installed)")
            return True
        else:
            print(f"❌ {package_name:<30} MISSING (required)")
            return False

def check_file(filepath, description):
    """Check if a file exists"""
    if os.path.exists(filepath):
        size_mb = os.path.getsize(filepath) / (1024 * 1024)
        print(f"✅ {description:<40} ({size_mb:.1f} MB)")
        return True
    else:
        print(f"❌ {description:<40} NOT FOUND")
        return False

def main():
    print("=" * 70)
    print("MIRAI SETUP VERIFICATION")
    print("=" * 70)
    
    all_good = True
    
    # Check Python version
    print("\n📌 PYTHON VERSION")
    py_version = sys.version_info
    if py_version >= (3, 9):
        print(f"✅ Python {py_version.major}.{py_version.minor}.{py_version.micro}")
    else:
        print(f"❌ Python {py_version.major}.{py_version.minor}.{py_version.micro} (requires 3.9+)")
        all_good = False
    
    # Check core packages
    print("\n📌 CORE PACKAGES (Required)")
    core_packages = [
        'openai',
        'tiktoken',
        'pandas',
        'numpy',
        'langchain',
        'langchain_core',
        'tqdm',
        'rank_bm25',
    ]
    
    for pkg in core_packages:
        if not check_package(pkg):
            all_good = False
    
    # Check optional packages
    print("\n📌 OPTIONAL PACKAGES (For Open-Source Models)")
    optional_packages = [
        'torch',
        'transformers',
        'vllm',
        'sentence_transformers',
        'sklearn',
    ]
    
    for pkg in optional_packages:
        check_package(pkg, optional=True)
    
    # Check OpenAI API key
    print("\n📌 ENVIRONMENT VARIABLES")
    api_key = os.environ.get('OPENAI_API_KEY')
    if api_key:
        print(f"✅ OPENAI_API_KEY              {'SET (hidden)'}")
    else:
        print(f"⚠️  OPENAI_API_KEY              NOT SET (required for OpenAI models)")
    
    # Check data files
    print("\n📌 DATA FILES")
    script_dir = Path(__file__).parent.resolve()
    data_dir = script_dir / "data" / "MIRAI"
    
    data_files = [
        (data_dir / "data_kg.csv", "Knowledge Graph data"),
        (data_dir / "data_news.csv", "News articles data"),
        (data_dir / "test_subset" / "relation_query.csv", "Test subset queries"),
        (data_dir / "test" / "relation_query.csv", "Full test queries"),
    ]
    
    for filepath, description in data_files:
        if not check_file(filepath, description):
            all_good = False
    
    # Check info files
    info_dir = script_dir / "data" / "info"
    info_files = [
        (info_dir / "dict_code2relation.json", "CAMEO code dictionary"),
        (info_dir / "dict_iso2alternames_GeoNames.json", "Country names dictionary"),
        (info_dir / "country_embeddings.npy", "Country embeddings"),
        (info_dir / "relation_embeddings.npy", "Relation embeddings"),
    ]
    
    for filepath, description in info_files:
        if not check_file(filepath, description):
            all_good = False
    
    # Summary
    print("\n" + "=" * 70)
    if all_good:
        print("✅ ALL CHECKS PASSED - Ready to run MIRAI!")
        print("\nNext steps:")
        print("  1. Set OPENAI_API_KEY if not already set:")
        print("     export OPENAI_API_KEY='your-api-key-here'")
        print("  2. Run example script:")
        print("     bash scripts/run_baseline_simpleRAG_working.sh")
        return 0
    else:
        print("❌ SOME CHECKS FAILED - Please fix issues above")
        print("\nTo install missing packages:")
        print("  pip install -r requirements.txt")
        print("\nTo download data:")
        print("  See README.md for Google Drive link")
        return 1

if __name__ == "__main__":
    sys.exit(main())
