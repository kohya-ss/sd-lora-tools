import json
import argparse
import sys
from safetensors import safe_open


# Relative imports or absolute imports
try:
    from .utils.common import setup_logging
except ImportError:
    # If not installed as a package
    import os

    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from sd_lora_tools.utils.common import setup_logging  # type: ignore

setup_logging()
import logging

logger = logging.getLogger(__name__)


parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, required=True)
args = parser.parse_args()

with safe_open(args.model, framework="pt") as f:
    metadata = f.metadata()

if metadata is None:
    logger.error("No metadata found")
else:
    # metadata is json dict, but not pretty printed
    # sort by key and pretty print
    print(json.dumps(metadata, indent=4, sort_keys=True))
