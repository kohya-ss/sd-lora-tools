"""Convert LoRA between sd-scripts and Diffusers formats."""

import argparse
import sys
from typing import Optional

import torch
from safetensors.torch import save_file

# Relative imports or absolute imports
try:
    from .utils.common import setup_logging, add_logging_arguments
    from .utils import safetensors_utils, metadata_utils
    from .utils.model_utils import str_to_dtype
except ImportError:
    # If not installed as a package
    import os

    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from sd_lora_tools.utils.common import setup_logging, add_logging_arguments  # type: ignore
    from sd_lora_tools.utils import safetensors_utils, metadata_utils  # type: ignore
    from sd_lora_tools.utils.model_utils import str_to_dtype  # type: ignore

setup_logging()
import logging

logger = logging.getLogger(__name__)

# Default prefixes
DEFAULT_SD_SCRIPTS_UNET_PREFIX = "lora_unet_"
DEFAULT_DIFFUSERS_UNET_PREFIX = "diffusion_model."

# Known prefixes in reference model files that should be stripped to get the base module name
REFERENCE_MODEL_PREFIXES_TO_STRIP = [
    "model.diffusion_model.",
    "diffusion_model.",
    "model.",
    "text_model.encoder.",
    "text_model.",
    "net.",  # Anima-preview
    "",  # fallback: no prefix
]


def split_scale(alpha: int, rank: int) -> tuple[float, float]:
    """Split alpha/rank into two factors for lora_down and lora_up scaling.

    Decomposes scale = alpha/rank = 2^k * (a/r) where a, r are odd.
    Splits 2^k evenly between down and up matrices.
    Non-power-of-2 remainder goes to one side only (up).

    Returns:
        (scale_down, scale_up) such that scale_down * scale_up == alpha / rank.
        scale_down is always an exact power of 2 (no rounding error).
    """
    a, r = int(alpha), int(rank)
    k = 0
    while a % 2 == 0:
        a //= 2
        k += 1
    while r % 2 == 0:
        r //= 2
        k -= 1
    # scale = 2^k * (a/r), a and r are both odd
    k1 = k // 2
    k2 = k - k1
    scale_down = 2.0**k1  # exact (power of 2)
    scale_up = 2.0**k2 * (a / r)  # rounding only if a != r
    return scale_down, scale_up


def build_underscore_to_dot_mapping(reference_model_paths: list[str]) -> dict[str, str]:
    """Build a mapping from underscore-delimited names to dot-delimited names
    using the reference model's key names.

    Only reads key names from the safetensors header (no tensor data is loaded).

    Args:
        reference_model_paths: Paths to reference model safetensors files.

    Returns:
        Dict mapping underscore_version -> dot_version.
        e.g. {"blocks_0_self_attn_to_q": "blocks.0.self_attn.to_q"}
    """
    mapping: dict[str, str] = {}
    for path in reference_model_paths:
        with safetensors_utils.MemoryEfficientSafeOpen(path) as f:
            for key in f.keys():
                # Try stripping each known prefix
                for prefix in REFERENCE_MODEL_PREFIXES_TO_STRIP:
                    if key.startswith(prefix):
                        candidate = key[len(prefix) :]
                        # Remove trailing .weight or .bias
                        for suffix in [".weight", ".bias"]:
                            if candidate.endswith(suffix):
                                candidate = candidate[: -len(suffix)]
                                break
                        underscore_version = candidate.replace(".", "_")
                        if underscore_version not in mapping:
                            mapping[underscore_version] = candidate
                        break  # use the first matching prefix
    return mapping


def detect_sd_scripts_te_prefixes(state_dict: dict[str, torch.Tensor]) -> list[str]:
    """Detect sd-scripts text encoder prefixes from keys.

    Returns:
        List of detected TE prefixes, e.g. ["lora_te_"] or ["lora_te1_", "lora_te2_"].
        Empty list if no TE keys found.
    """
    prefixes: list[str] = []
    # Check numbered TEs first (lora_te1_, lora_te2_, lora_te3_, ...)
    for i in range(1, 10):
        prefix = f"lora_te{i}_"
        if any(k.startswith(prefix) for k in state_dict):
            prefixes.append(prefix)
    if prefixes:
        return prefixes
    # Check single TE
    if any(k.startswith("lora_te_") for k in state_dict):
        return ["lora_te_"]
    return []


def detect_diffusers_te_prefixes(state_dict: dict[str, torch.Tensor]) -> list[str]:
    """Detect Diffusers text encoder prefixes from keys.

    Returns:
        List of detected TE prefixes, e.g. ["text_encoder."] or ["text_encoder.", "text_encoder_2."].
        Empty list if no TE keys found.
    """
    prefixes: list[str] = []
    # Check numbered TEs (text_encoder_2., text_encoder_3., ...)
    has_numbered = False
    for i in range(2, 10):
        prefix = f"text_encoder_{i}."
        if any(k.startswith(prefix) for k in state_dict):
            prefixes.append(prefix)
            has_numbered = True
    # Check base TE
    if any(k.startswith("text_encoder.") for k in state_dict):
        prefixes.insert(0, "text_encoder.")
    if prefixes:
        return prefixes
    return []


def get_default_diffusers_te_prefixes(count: int) -> list[str]:
    """Get default Diffusers TE prefixes for a given count.

    Args:
        count: Number of text encoders.

    Returns:
        e.g. ["text_encoder."] for 1, ["text_encoder.", "text_encoder_2."] for 2.
    """
    if count == 0:
        return []
    if count == 1:
        return ["text_encoder."]
    return ["text_encoder."] + [f"text_encoder_{i}." for i in range(2, count + 1)]


def get_default_sd_scripts_te_prefixes(count: int) -> list[str]:
    """Get default sd-scripts TE prefixes for a given count.

    Args:
        count: Number of text encoders.

    Returns:
        e.g. ["lora_te_"] for 1, ["lora_te1_", "lora_te2_"] for 2.
    """
    if count == 0:
        return []
    if count == 1:
        return ["lora_te_"]
    return [f"lora_te{i}_" for i in range(1, count + 1)]


def convert_sd_scripts_to_diffusers(
    state_dict: dict[str, torch.Tensor],
    mapping: dict[str, str],
    sd_scripts_unet_prefix: str,
    sd_scripts_te_prefixes: list[str],
    diffusers_unet_prefix: str,
    diffusers_te_prefixes: list[str],
) -> dict[str, torch.Tensor]:
    """Convert sd-scripts format LoRA to Diffusers format.

    Remaps key names, converts lora_down/lora_up to lora_A/lora_B,
    and bakes alpha scaling into weights.
    """
    # Build prefix mapping: sd-scripts prefix -> diffusers prefix
    prefix_map: dict[str, str] = {sd_scripts_unet_prefix: diffusers_unet_prefix}
    for sd_te, diff_te in zip(sd_scripts_te_prefixes, diffusers_te_prefixes):
        prefix_map[sd_te] = diff_te

    all_sd_prefixes = [sd_scripts_unet_prefix] + sd_scripts_te_prefixes

    # Pass 1: Collect modules (alpha, down, up per lora_name)
    modules: dict[str, dict[str, torch.Tensor]] = {}  # lora_name -> {"alpha": ..., "down": ..., "up": ...}
    for key, weight in state_dict.items():
        # Find matching prefix
        matched_prefix = None
        for pfx in all_sd_prefixes:
            if key.startswith(pfx):
                matched_prefix = pfx
                break
        if matched_prefix is None:
            logger.warning(f"Skipping key with unknown prefix: {key}")
            continue

        # Split: prefix + lora_name_without_prefix + "." + suffix_part
        lora_name = key.split(".")[0]  # everything before the first dot
        suffix_part = key[len(lora_name) + 1 :]  # after the first dot

        if lora_name not in modules:
            modules[lora_name] = {"prefix": matched_prefix}
        if suffix_part == "alpha":
            modules[lora_name]["alpha"] = weight
        elif "lora_down" in suffix_part:
            modules[lora_name]["down"] = weight
        elif "lora_up" in suffix_part:
            modules[lora_name]["up"] = weight
        else:
            logger.warning(f"Skipping key with unknown suffix: {key}")

    # Pass 2: Convert each module
    new_state_dict: dict[str, torch.Tensor] = {}
    unmapped_count = 0
    for lora_name, module in modules.items():
        if "down" not in module or "up" not in module:
            logger.warning(f"Skipping incomplete module: {lora_name}")
            continue

        matched_prefix = module["prefix"]
        diffusers_prefix = prefix_map[matched_prefix]
        underscore_base = lora_name[len(matched_prefix) :]

        # Look up dot-delimited name from mapping
        if underscore_base in mapping:
            dot_base = mapping[underscore_base]
        else:
            # Fallback: naive underscore to dot replacement
            dot_base = underscore_base.replace("_", ".")
            unmapped_count += 1
            logger.warning(f"Key not found in reference model, using fallback: {underscore_base} -> {dot_base}")

        down_weight = module["down"]
        up_weight = module["up"]
        rank = down_weight.shape[0]
        alpha = int(module["alpha"].item()) if "alpha" in module else rank

        # Scale weights by alpha
        if alpha != rank:
            scale_down, scale_up = split_scale(alpha, rank)
            down_weight = down_weight * scale_down
            up_weight = up_weight * scale_up

        new_state_dict[f"{diffusers_prefix}{dot_base}.lora_A.weight"] = down_weight
        new_state_dict[f"{diffusers_prefix}{dot_base}.lora_B.weight"] = up_weight

    if unmapped_count > 0:
        logger.warning(f"{unmapped_count} key(s) not found in reference model mapping (used fallback)")

    return new_state_dict


def convert_diffusers_to_sd_scripts(
    state_dict: dict[str, torch.Tensor],
    sd_scripts_unet_prefix: str,
    sd_scripts_te_prefixes: list[str],
    diffusers_unet_prefix: str,
    diffusers_te_prefixes: list[str],
) -> dict[str, torch.Tensor]:
    """Convert Diffusers format LoRA to sd-scripts format.

    Remaps key names, converts lora_A/lora_B to lora_down/lora_up,
    and adds alpha = rank for each module.
    """
    # Build prefix mapping: diffusers prefix -> sd-scripts prefix
    prefix_map: dict[str, str] = {diffusers_unet_prefix: sd_scripts_unet_prefix}
    for diff_te, sd_te in zip(diffusers_te_prefixes, sd_scripts_te_prefixes):
        prefix_map[diff_te] = sd_te

    all_diff_prefixes = [diffusers_unet_prefix] + diffusers_te_prefixes

    new_state_dict: dict[str, torch.Tensor] = {}
    lora_dims: dict[str, int] = {}  # lora_name -> rank

    for key, weight in state_dict.items():
        # Find matching prefix
        matched_prefix = None
        for pfx in all_diff_prefixes:
            if key.startswith(pfx):
                matched_prefix = pfx
                break
        if matched_prefix is None:
            logger.warning(f"Skipping key with unknown prefix: {key}")
            continue

        # Determine suffix mapping
        remaining = key[len(matched_prefix) :]
        if remaining.endswith(".lora_A.weight"):
            base = remaining[: -len(".lora_A.weight")]
            sd_suffix = ".lora_down.weight"
            rank = weight.shape[0]
        elif remaining.endswith(".lora_B.weight"):
            base = remaining[: -len(".lora_B.weight")]
            sd_suffix = ".lora_up.weight"
            rank = weight.shape[1]
        elif remaining.endswith(".alpha"):
            # Diffusers alpha (rare but possible) — skip, we will add our own
            continue
        else:
            logger.warning(f"Skipping key with unknown suffix: {key}")
            continue

        sd_scripts_prefix = prefix_map[matched_prefix]
        underscore_base = base.replace(".", "_")
        lora_name = f"{sd_scripts_prefix}{underscore_base}"

        new_state_dict[f"{lora_name}{sd_suffix}"] = weight

        # Track rank for alpha
        if lora_name not in lora_dims:
            lora_dims[lora_name] = rank

    # Add alpha = rank for each module
    for lora_name, rank in lora_dims.items():
        new_state_dict[f"{lora_name}.alpha"] = torch.tensor(rank)

    return new_state_dict


def convert(args: argparse.Namespace) -> None:
    """Main conversion orchestration."""
    # Validate
    if not args.input.lower().endswith(".safetensors"):
        logger.error("Input file must be a .safetensors file")
        sys.exit(1)
    if not args.output.lower().endswith(".safetensors"):
        logger.error("Output file must be a .safetensors file")
        sys.exit(1)
    if args.target == "diffusers" and not args.reference_model:
        logger.error("--reference_model is required for sd-scripts -> Diffusers conversion")
        sys.exit(1)

    save_dtype = str_to_dtype(args.save_precision) if args.save_precision else None

    # Load input LoRA
    logger.info(f"Loading LoRA: {args.input}")
    state_dict, metadata = safetensors_utils.load_safetensors_without_mmap(args.input)

    # Resolve prefixes
    sd_scripts_unet_prefix = args.sd_scripts_unet_prefix or DEFAULT_SD_SCRIPTS_UNET_PREFIX
    diffusers_unet_prefix = args.diffusers_unet_prefix or DEFAULT_DIFFUSERS_UNET_PREFIX

    if args.target == "diffusers":
        # Detect TE prefixes from sd-scripts keys
        if args.sd_scripts_te_prefix is not None and len(args.sd_scripts_te_prefix) > 0:
            sd_scripts_te_prefixes = args.sd_scripts_te_prefix
        else:
            sd_scripts_te_prefixes = detect_sd_scripts_te_prefixes(state_dict)

        if args.diffusers_te_prefix is not None and len(args.diffusers_te_prefix) > 0:
            diffusers_te_prefixes = args.diffusers_te_prefix
        else:
            diffusers_te_prefixes = get_default_diffusers_te_prefixes(len(sd_scripts_te_prefixes))

        if len(sd_scripts_te_prefixes) != len(diffusers_te_prefixes):
            logger.error(
                f"Mismatch between sd-scripts TE prefixes ({len(sd_scripts_te_prefixes)}) "
                f"and Diffusers TE prefixes ({len(diffusers_te_prefixes)})"
            )
            sys.exit(1)

        logger.info(f"UNet prefix: {sd_scripts_unet_prefix} -> {diffusers_unet_prefix}")
        for sd_te, diff_te in zip(sd_scripts_te_prefixes, diffusers_te_prefixes):
            logger.info(f"TE prefix: {sd_te} -> {diff_te}")

        # Build reference model mapping
        logger.info(f"Loading reference model keys from {len(args.reference_model)} file(s)")
        mapping = build_underscore_to_dot_mapping(args.reference_model)
        logger.info(f"Built mapping with {len(mapping)} entries")

        new_state_dict = convert_sd_scripts_to_diffusers(
            state_dict, mapping, sd_scripts_unet_prefix, sd_scripts_te_prefixes, diffusers_unet_prefix, diffusers_te_prefixes
        )
    else:
        # target == "sd_scripts"
        # Detect TE prefixes from Diffusers keys
        if args.diffusers_te_prefix is not None and len(args.diffusers_te_prefix) > 0:
            diffusers_te_prefixes = args.diffusers_te_prefix
        else:
            diffusers_te_prefixes = detect_diffusers_te_prefixes(state_dict)

        if args.sd_scripts_te_prefix is not None and len(args.sd_scripts_te_prefix) > 0:
            sd_scripts_te_prefixes = args.sd_scripts_te_prefix
        else:
            sd_scripts_te_prefixes = get_default_sd_scripts_te_prefixes(len(diffusers_te_prefixes))

        if len(sd_scripts_te_prefixes) != len(diffusers_te_prefixes):
            logger.error(
                f"Mismatch between sd-scripts TE prefixes ({len(sd_scripts_te_prefixes)}) "
                f"and Diffusers TE prefixes ({len(diffusers_te_prefixes)})"
            )
            sys.exit(1)

        logger.info(f"UNet prefix: {diffusers_unet_prefix} -> {sd_scripts_unet_prefix}")
        for diff_te, sd_te in zip(diffusers_te_prefixes, sd_scripts_te_prefixes):
            logger.info(f"TE prefix: {diff_te} -> {sd_te}")

        new_state_dict = convert_diffusers_to_sd_scripts(
            state_dict, sd_scripts_unet_prefix, sd_scripts_te_prefixes, diffusers_unet_prefix, diffusers_te_prefixes
        )

    # Cast to save_dtype if specified
    if save_dtype is not None:
        for key in new_state_dict:
            v = new_state_dict[key]
            if isinstance(v, torch.Tensor) and v.is_floating_point() and v.dtype != save_dtype:
                new_state_dict[key] = v.to(save_dtype)

    # Update metadata
    if metadata is None:
        metadata = {}
    metadata_utils.update_metadata_hashes(new_state_dict, metadata)

    # Save
    logger.info(f"Saving to: {args.output}")
    save_file(new_state_dict, args.output, metadata=metadata)
    logger.info(f"Done. {len(new_state_dict)} keys written.")


def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert LoRA between sd-scripts and Diffusers formats")
    parser.add_argument("--input", type=str, required=True, help="Input LoRA safetensors file")
    parser.add_argument("--output", type=str, required=True, help="Output safetensors file")
    parser.add_argument(
        "--target",
        type=str,
        required=True,
        choices=["sd_scripts", "diffusers"],
        help="Target format to convert to",
    )
    parser.add_argument(
        "--reference_model",
        type=str,
        nargs="+",
        default=None,
        help="Reference model safetensors file(s) for key name resolution "
        "(required for sd-scripts -> Diffusers conversion). "
        "Multiple files can be specified if UNet and Text Encoder weights are in separate files.",
    )

    # Prefix overrides
    parser.add_argument(
        "--sd_scripts_unet_prefix",
        type=str,
        default=None,
        help=f"sd-scripts UNet/DiT prefix (default: {DEFAULT_SD_SCRIPTS_UNET_PREFIX})",
    )
    parser.add_argument(
        "--sd_scripts_te_prefix",
        type=str,
        nargs="*",
        default=None,
        help="sd-scripts text encoder prefix(es) (default: auto-detect from keys)",
    )
    parser.add_argument(
        "--diffusers_unet_prefix",
        type=str,
        default=None,
        help=f"Diffusers UNet/DiT prefix (default: {DEFAULT_DIFFUSERS_UNET_PREFIX})",
    )
    parser.add_argument(
        "--diffusers_te_prefix",
        type=str,
        nargs="*",
        default=None,
        help="Diffusers text encoder prefix(es) (default: auto-detect from keys)",
    )

    parser.add_argument(
        "--save_precision",
        type=str,
        default=None,
        choices=[None, "float", "fp16", "bf16"],
        help="Precision for saving (default: same as input)",
    )

    return parser


def main():
    parser = setup_parser()
    add_logging_arguments(parser)
    args = parser.parse_args()
    setup_logging(args, reset=True)
    convert(args)


if __name__ == "__main__":
    main()
