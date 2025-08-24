import argparse
import os
import re
import sys
from typing import Dict, Optional
import torch
from tqdm import tqdm


# Relative imports or absolute imports
try:
    from .utils.safetensors_utils import MemoryEfficientSafeOpen, mem_eff_save_file
    from .utils.common import setup_logging, add_logging_arguments
    from .utils.model_utils import str_to_dtype
except ImportError:
    # If not installed as a package
    import os

    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from sd_lora_tools.utils.safetensors_utils import MemoryEfficientSafeOpen, mem_eff_save_file  # type: ignore
    from sd_lora_tools.utils.common import setup_logging, add_logging_arguments  # type: ignore
    from sd_lora_tools.utils.model_utils import str_to_dtype  # type: ignore

setup_logging()
import logging

logger = logging.getLogger(__name__)

import torch


def merge_single_lora_weight(
    weight: torch.Tensor, lora_down: torch.Tensor, lora_up: torch.Tensor, alpha: float, ratio: float
) -> torch.Tensor:
    rank = lora_down.size(0)
    ratio = ratio * (alpha / rank)  # Adjust ratio by alpha and rank

    # Merge weights
    # W <- W + U * D
    if len(weight.size()) == 2:
        # linear
        if len(lora_up.size()) == 4:  # use linear projection mismatch
            lora_up = lora_up.squeeze(3).squeeze(2)
            lora_down = lora_down.squeeze(3).squeeze(2)
        weight = weight + ratio * (lora_up @ lora_down)
    elif lora_down.size()[2:4] == (1, 1):
        # conv2d 1x1
        weight = weight + ratio * (lora_up.squeeze(3).squeeze(2) @ lora_down.squeeze(3).squeeze(2)).unsqueeze(2).unsqueeze(3)
    else:
        # conv2d 3x3
        conved = torch.nn.functional.conv2d(lora_down.permute(1, 0, 2, 3), lora_up).permute(1, 0, 2, 3)
        weight = weight + ratio * conved
    return weight


def merge_lora_weights(
    key: str,
    base_weight: torch.Tensor,
    lora_f_list: list[MemoryEfficientSafeOpen],
    lora_keys_list: list[list[str]],
    ratios: list[float],
    re_scales: Optional[list[tuple[re.Pattern, float]]],
    merge_dtype: torch.dtype,
) -> tuple[list[bool], torch.Tensor]:
    if not key.endswith(".weight"):
        return [False] * len(lora_f_list), base_weight
    
    # Remove annoying prefix from the base model key
    if key.startswith("model."):
        key = key[len("model.") :]
    if key.startswith("diffusion_model."):
        key = key[len("diffusion_model.") :]
    if key.startswith("text_encoder."):
        key = key[len("text_encoder.") :]

    # Create a list of all possible LoRA prefixes
    prefixes = ["lora_unet_", "lora_te_", "lora_te1_", "lora_te2_", "lora_te3_"]

    lora_name_without_prefix = key[: -len(".weight")].replace(".", "_")  # e.g. "down_blocks_0_attentions_0_to_k_proj"
    success_list = []
    for lora_f, lora_keys, ratio in zip(lora_f_list, lora_keys_list, ratios):
        success = False
        for prefix in prefixes:
            lora_module_name = prefix + lora_name_without_prefix
            lora_key = lora_module_name + ".lora_down.weight"

            if lora_key in lora_keys:
                # merge weights
                lora_down = lora_f.get_tensor(lora_key).to(dtype=merge_dtype)
                lora_up = lora_f.get_tensor(lora_key.replace("lora_down", "lora_up")).to(dtype=merge_dtype)
                alpha_key = lora_module_name + ".alpha"

                if alpha_key in lora_keys:
                    alpha = lora_f.get_tensor(alpha_key).to(dtype=merge_dtype)
                else:
                    alpha = lora_down.size(0)  # same as rank(dim)
                alpha = float(alpha)

                # calculate re_scales
                if re_scales:
                    scale_by_regex = None
                    for re_i, scale_i in re_scales:
                        if re_i.search(lora_module_name):
                            scale_by_regex = scale_i
                            break
                    if scale_by_regex is not None:
                        ratio *= scale_by_regex

                base_weight = merge_single_lora_weight(base_weight, lora_down, lora_up, alpha, ratio)

                logger.debug(f"merged {lora_module_name} with alpha {alpha} and ratio {ratio}")
                success = True
                break  # break prefix loop

        # end of prefix loop, process next lora
        success_list.append(success)

    return success_list, base_weight


def load_model_keys_and_metadata(model_path: str) -> tuple[list[str], Optional[Dict[str, str]]]:
    with MemoryEfficientSafeOpen(model_path) as f:
        keys = list(f.keys())
        metadata = f.metadata()
    return keys, metadata


def merge(args):
    assert len(args.models) == len(
        args.ratios
    ), f"number of models must be equal to number of ratios / モデルの数と重みの数は合わせてください"

    re_scales = None
    if args.regex_scales:
        # compile str to regex
        re_scales = []
        for regex_and_scale in args.regex_scales:
            regex, scale = regex_and_scale.rsplit("=", 1)
            try:
                re_obj = re.compile(regex)
                re_scales.append((re_obj, float(scale)))
            except re.error as e:
                logger.error(f"Invalid regex: {regex}, error: {e}")
                sys.exit(1)

    merge_dtype = str_to_dtype(args.precision)
    save_dtype = str_to_dtype(args.save_precision)
    if save_dtype is None:
        save_dtype = merge_dtype

    # enumerate keys for all models
    logger.info("Checking model keys...")
    base_model_keys, base_model_metadata = load_model_keys_and_metadata(args.base_model)
    lora_keys_list = [load_model_keys_and_metadata(model)[0] for model in args.models]

    # count modules for each LoRA
    module_count = [0] * len(args.models)
    for i, lora_keys in enumerate(lora_keys_list):
        module_count[i] = len([k for k in lora_keys if k.endswith("lora_down.weight")])

    # on the fly merging
    merged_sd = {}
    with MemoryEfficientSafeOpen(args.base_model) as base_f:
        # open each LoRA model
        lora_f_list = [MemoryEfficientSafeOpen(model) for model in args.models]
        for key in tqdm(base_model_keys):
            value = base_f.get_tensor(key).to(dtype=merge_dtype)
            success_list, value = merge_lora_weights(key, value, lora_f_list, lora_keys_list, args.ratios, re_scales, merge_dtype)  # type: ignore
            merged_sd[key] = value.to(device="cpu", dtype=save_dtype)

            # decrement module count if successful
            for i, success in enumerate(success_list):
                if success:
                    module_count[i] -= 1

    # verify module count
    if any(count > 0 for count in module_count):
        logger.warning(f"Some LoRA modules were not merged successfully: {module_count}")
    else:
        logger.info("All LoRA modules merged successfully.")

    # save merged state dict
    logger.info(f"saving merged model to: {args.save_to}")

    save_dir = os.path.dirname(args.save_to)
    os.makedirs(save_dir, exist_ok=True)

    mem_eff_save_file(merged_sd, args.save_to, metadata=base_model_metadata)


def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save_precision",
        type=str,
        default=None,
        choices=[None, "float", "fp16", "bf16"],
        help="precision in saving, same to merging if omitted / 保存時に精度を変更して保存する、省略時はマージ時の精度と同じ",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="float",
        choices=["float", "fp16", "bf16"],
        help="precision in merging (float is recommended) / マージの計算時の精度（floatを推奨）",
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default=None,
        required=True,
        help="base model to use for merging: safetensors file / マージに使用するベースモデル、safetensorsのみ対応",
    )
    parser.add_argument(
        "--save_to",
        type=str,
        default=None,
        required=True,
        help="destination file name: safetensors file / 保存先のファイル名、safetensorsのみ対応",
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        help="LoRA models to merge: safetensors file / マージするLoRAモデル、safetensorsのみ対応",
    )
    parser.add_argument("--ratios", type=float, nargs="+", help="ratios for each model / それぞれのLoRAモデルの比率")
    parser.add_argument(
        "--regex_scales",
        type=str,
        nargs="*",
        help="scale for layers (modules) like LBW / LBWのような層（モジュール）ごとのスケール、正規表現で指定 "
        + "e.g. '(up|down)_blocks_([012])_(resnets|upsamplers|downsamplers|attentions)_(\\d+)_=0.5'",
    )
    parser.add_argument(
        "--device", type=str, default=None, help="device to use, cuda for GPU / 計算を行うデバイス、cuda でGPUを使う"
    )

    return parser


def main():
    parser = setup_parser()
    add_logging_arguments(parser)
    args = parser.parse_args()
    merge(args)


if __name__ == "__main__":
    main()
