"""interlingua: bottom-up conlang extraction from SAE features.

WARNING: HF_HUB_CACHE is set via os.environ.setdefault below, but this only
takes effect if THIS module is imported BEFORE huggingface_hub / transformers.
For standalone scripts, set HF_HUB_CACHE in the shell or at the top of the
script before any HF imports. See external/run_crystal_pca.py for the pattern.
"""

import os
from pathlib import Path

__version__ = "0.1.0"

FAUNA_ROOT = Path("/media/menser/fauna/interlingua")
HF_CACHE_DIR = FAUNA_ROOT / "hf-cache"
DATA_DIR = FAUNA_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"

os.environ.setdefault("HF_HUB_CACHE", str(HF_CACHE_DIR))
