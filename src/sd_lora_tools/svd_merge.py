import argparse
import os
import re
import sys
from typing import Optional
import torch
from safetensors.torch import save_file
from tqdm import tqdm

# Relative imports or absolute imports
try:
    from .utils.common import setup_logging, add_logging_arguments
    from .utils import metadata_utils
    from .utils.model_utils import str_to_dtype, LoRASaverLoader
except ImportError:
    # If not installed as a package
    import os

    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from sd_lora_tools.utils.common import setup_logging, add_logging_arguments  # type: ignore
    from sd_lora_tools.utils import metadata_utils  # type: ignore
    from sd_lora_tools.utils.model_utils import str_to_dtype, LoRASaverLoader  # type: ignore

setup_logging()
import logging

logger = logging.getLogger(__name__)


# clamping is disabled
# CLAMP_QUANTILE = 0.99


def merge_lora_models(
    lora_loader: LoRASaverLoader,
    models,
    ratios,
    re_scales,
    new_rank,
    new_conv_rank,
    device: torch.device,
    merge_dtype,
    use_svd_lowrank: bool,
) -> tuple[dict[str, torch.Tensor], list[Optional[dict[str, str]]]]:
    logger.info(f"new rank: {new_rank}, new conv rank: {new_conv_rank}")
    merged_sd = {}

    metadatas = []
    for model, ratio in zip(models, ratios):
        logger.info(f"loading: {model}")
        lora_sd, metadata = lora_loader.load(model, merge_dtype)
        metadatas.append(metadata)

        # merge
        logger.info(f"merging...")
        for key in tqdm(list(lora_sd.keys())):
            if "lora_down" not in key:
                continue

            lora_module_name = key[: key.rfind(".lora_down")]

            down_weight = lora_sd[key]
            network_dim = down_weight.size()[0]

            up_weight = lora_sd[lora_module_name + ".lora_up.weight"]
            alpha = lora_sd.get(lora_module_name + ".alpha", torch.tensor(network_dim))

            in_dim = down_weight.size()[1]
            out_dim = up_weight.size()[0]
            conv2d = len(down_weight.size()) == 4
            kernel_size = None if not conv2d else down_weight.size()[2:4]
            # logger.info(lora_module_name, network_dim, alpha, in_dim, out_dim, kernel_size)

            # make original weight if not exist
            if lora_module_name not in merged_sd:
                weight = torch.zeros((out_dim, in_dim, *kernel_size) if conv2d else (out_dim, in_dim), dtype=merge_dtype)  # type: ignore
            else:
                weight = merged_sd[lora_module_name]
            if device:
                weight = weight.to(device)

            # merge to weight
            if device:
                up_weight = up_weight.to(device)
                down_weight = down_weight.to(device)

            # W <- W + U * D
            scale = alpha / network_dim

            if re_scales:
                scale_by_regex = None
                for re_i, scale_i in re_scales:
                    if re_i.search(lora_module_name):
                        scale_by_regex = scale_i
                        break
                if scale_by_regex is not None:
                    scale *= scale_by_regex

            if device:  # and isinstance(scale, torch.Tensor):
                scale = scale.to(device)

            if not conv2d:  # linear
                weight = weight + ratio * (up_weight @ down_weight) * scale
            elif kernel_size == (1, 1):
                weight = (
                    weight
                    + ratio
                    * (up_weight.squeeze(3).squeeze(2) @ down_weight.squeeze(3).squeeze(2)).unsqueeze(2).unsqueeze(3)
                    * scale
                )
            else:
                conved = torch.nn.functional.conv2d(down_weight.permute(1, 0, 2, 3), up_weight).permute(1, 0, 2, 3)
                weight = weight + ratio * conved * scale

            merged_sd[lora_module_name] = weight.to("cpu")

    # extract from merged weights
    logger.info("extract new lora...")
    merged_lora_sd = {}
    with torch.no_grad():
        for lora_module_name, mat in tqdm(list(merged_sd.items())):
            if device:
                mat = mat.to(device)

            conv2d = len(mat.size()) == 4
            kernel_size = None if not conv2d else mat.size()[2:4]
            conv2d_3x3 = conv2d and kernel_size != (1, 1)
            out_dim, in_dim = mat.size()[0:2]

            if conv2d:
                if conv2d_3x3:
                    mat = mat.flatten(start_dim=1)
                else:
                    mat = mat.squeeze()

            module_new_rank = new_conv_rank if conv2d_3x3 else new_rank
            module_new_rank = min(module_new_rank, in_dim, out_dim)  # LoRA rank cannot exceed the original dim

            if use_svd_lowrank and module_new_rank < min(mat.size()) // 4:  # new_module_rank is actually low-rank
                U, S, Vh = torch.svd_lowrank(mat, q=module_new_rank)  # , niter=1)
                U = U @ torch.diag(S)
                Vh = Vh.T
            else:
                U, S, Vh = torch.linalg.svd(mat)
                U = U[:, :module_new_rank]
                S = S[:module_new_rank]
                U = U @ torch.diag(S)

                Vh = Vh[:module_new_rank, :]

            # dist = torch.cat([U.flatten(), Vh.flatten()])
            # hi_val = torch.quantile(dist, CLAMP_QUANTILE)
            # low_val = -hi_val
            # U = U.clamp(low_val, hi_val)
            # Vh = Vh.clamp(low_val, hi_val)

            if conv2d:
                U = U.reshape(out_dim, module_new_rank, 1, 1)
                Vh = Vh.reshape(module_new_rank, in_dim, kernel_size[0], kernel_size[1])  # type: ignore

            up_weight = U
            down_weight = Vh

            merged_lora_sd[lora_module_name + ".lora_up.weight"] = up_weight.to("cpu").contiguous()
            merged_lora_sd[lora_module_name + ".lora_down.weight"] = down_weight.to("cpu").contiguous()
            merged_lora_sd[lora_module_name + ".alpha"] = torch.tensor(module_new_rank, device="cpu")

    return merged_lora_sd, metadatas


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

    lora_saver_loader = LoRASaverLoader()
    new_conv_rank = args.new_conv_rank if args.new_conv_rank is not None else args.new_rank
    device = torch.device(args.device) if args.device else torch.device("cpu")
    state_dict, metadatas = merge_lora_models(
        lora_saver_loader,
        args.models,
        args.ratios,
        re_scales,
        args.new_rank,
        new_conv_rank,
        device,
        merge_dtype,
        args.use_svd_lowrank,
    )

    # use first not None metadata
    metadata = next((m for m in metadatas if m is not None), None)

    # if metadata is not None, update metadata
    if metadata is not None:
        dims = f"{args.new_rank}"
        alphas = f"{args.new_rank}"
        if new_conv_rank is not None:
            network_args = {"conv_dim": new_conv_rank, "conv_alpha": new_conv_rank}
        else:
            network_args = None
        metadata_utils.update_metadata_dim_alpha(metadata, dims, alphas, network_args)

    # cast to save_dtype before calculating hashes
    for key in list(state_dict.keys()):
        value = state_dict[key]
        if type(value) == torch.Tensor and value.dtype.is_floating_point and value.dtype != save_dtype:
            state_dict[key] = value.to(save_dtype)

    if metadata is not None:
        logger.info(f"calculating hashes and creating metadata...")
        metadata_utils.update_metadata_hashes(state_dict, metadata)
        if not args.no_metadata:
            metadata_utils.update_merged_from(metadata, metadatas, args.models)
            metadata_utils.update_title(metadata, os.path.splitext(os.path.basename(args.save_to))[0])

    logger.info(f"saving model to: {args.save_to}")
    lora_saver_loader.save(args.save_to, state_dict, metadata)


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
        "--save_to",
        type=str,
        default=None,
        help="destination file name: ckpt or safetensors file / 保存先のファイル名、ckptまたはsafetensors",
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="*",
        help="LoRA models to merge: ckpt or safetensors file / マージするLoRAモデル、ckptまたはsafetensors",
    )
    parser.add_argument("--ratios", type=float, nargs="*", help="ratios for each model / それぞれのLoRAモデルの比率")
    parser.add_argument(
        "--regex_scales",
        type=str,
        nargs="*",
        help="scale for layers (modules) like LBW / LBWのような層（モジュール）ごとのスケール、正規表現で指定 "
        + "e.g. '(up|down)_blocks_([012])_(resnets|upsamplers|downsamplers|attentions)_(\\d+)_=0.5'",
    )
    parser.add_argument("--new_rank", type=int, default=4, help="Specify rank of output LoRA / 出力するLoRAのrank (dim)")
    parser.add_argument(
        "--new_conv_rank",
        type=int,
        default=None,
        help="Specify rank of output LoRA for Conv2d 3x3, None for same as new_rank / 出力するConv2D 3x3 LoRAのrank (dim)、Noneでnew_rankと同じ",
    )
    parser.add_argument(
        "--device", type=str, default=None, help="device to use, cuda for GPU / 計算を行うデバイス、cuda でGPUを使う"
    )
    parser.add_argument(
        "--use_svd_lowrank",
        action="store_true",
        help="use torch.svd_lowrank for merging / マージ時にtorch.svd_lowrankを使用する",
    )
    parser.add_argument(
        "--no_metadata",
        action="store_true",
        help="do not save sai modelspec merged_from metadata (minimum ss_metadata for LoRA is saved) / "
        + "sai modelspecのmerged_fromメタデータを保存しない（LoRAの最低限のss_metadataは保存される）",
    )

    return parser


def main():
    parser = setup_parser()
    add_logging_arguments(parser)
    args = parser.parse_args()
    merge(args)


if __name__ == "__main__":
    main()
