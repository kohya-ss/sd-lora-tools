import argparse
import os
import torch
from safetensors.torch import load_file

# Relative imports or absolute imports
try:
    from .utils.common import setup_logging
except ImportError:
    # If not installed as a package
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from sd_lora_tools.utils.common import setup_logging  # type: ignore

setup_logging()
import logging

logger = logging.getLogger(__name__)


def main(file):
    logger.info(f"loading: {file}")
    if os.path.splitext(file)[1] == ".safetensors":
        sd = load_file(file)
    else:
        sd = torch.load(file, map_location="cpu")

    key_values = []

    keys = list(sd.keys())
    for key in keys:
        if "lora_up" in key or "lora_down" in key or "lora_A" in key or "lora_B" in key or "oft_" in key:
            key_values.append((key, sd[key]))
    print(f"number of LoRA modules: {len(key_values)}")

    if args.show_all_keys:
        for key in [k for k in keys if k not in key_values]:
            key_values.append((key, sd[key]))
        print(f"number of all modules: {len(key_values)}")

    for key, value in key_values:
        value = value.to(torch.float32)
        print(f"{key},{str(tuple(value.size())).replace(', ', '-')},{torch.mean(torch.abs(value))},{torch.min(torch.abs(value))}")


def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("file", type=str, help="model file to check / 重みを確認するモデルファイル")
    parser.add_argument("-s", "--show_all_keys", action="store_true", help="show all keys / 全てのキーを表示する")

    return parser


if __name__ == "__main__":
    parser = setup_parser()

    args = parser.parse_args()

    main(args.file)
