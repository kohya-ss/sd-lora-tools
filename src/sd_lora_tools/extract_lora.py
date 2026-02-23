"""Extract LoRA from the difference between original and tuned model weights via SVD.

This is a universal extraction tool that works directly on safetensors weight files
without requiring model instantiation, making it architecture-agnostic.
"""

import argparse
import json
import os
import sys
from typing import Optional

import torch
from safetensors.torch import save_file
from tqdm import tqdm

# Relative imports or absolute imports
try:
    from .utils.common import setup_logging, add_logging_arguments
    from .utils import safetensors_utils, metadata_utils
    from .utils.model_utils import str_to_dtype, REFERENCE_MODEL_PREFIXES_TO_STRIP
except ImportError:
    # If not installed as a package
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from sd_lora_tools.utils.common import setup_logging, add_logging_arguments  # type: ignore
    from sd_lora_tools.utils import safetensors_utils, metadata_utils  # type: ignore
    from sd_lora_tools.utils.model_utils import str_to_dtype, REFERENCE_MODEL_PREFIXES_TO_STRIP  # type: ignore

setup_logging()
import logging

logger = logging.getLogger(__name__)


# --- Prefix Detection ---


def detect_model_prefix(keys: list[str], known_prefixes: list[str]) -> str:
    """Detect and return the common prefix of model weight keys.

    Tries each known prefix in order. A prefix matches if the majority of
    weight-like keys (ending with .weight or .bias) start with it.

    Args:
        keys: List of keys from the model file.
        known_prefixes: List of known prefixes to try, in priority order.

    Returns:
        The detected prefix string, or "" if no prefix matches.
    """
    # Filter to weight/bias keys only (ignore metadata-like keys)
    weight_keys = [k for k in keys if k.endswith(".weight") or k.endswith(".bias")]
    if not weight_keys:
        return ""

    for prefix in known_prefixes:
        if prefix == "":
            continue  # skip empty fallback during detection
        matching = sum(1 for k in weight_keys if k.startswith(prefix))
        if matching == len(weight_keys):
            return prefix

    # Try majority vote as fallback
    for prefix in known_prefixes:
        if prefix == "":
            continue
        matching = sum(1 for k in weight_keys if k.startswith(prefix))
        if matching > len(weight_keys) * 0.5:
            logger.warning(f"Using prefix '{prefix}' based on majority ({matching}/{len(weight_keys)} keys)")
            return prefix

    return ""


# --- Key Transform ---


def make_lora_key(model_key: str, model_prefix: str, lora_prefix: str) -> str:
    """Convert a model weight key to an sd-scripts LoRA key base name.

    Steps:
        1. Strip model_prefix from the key
        2. Strip trailing '.weight'
        3. Replace '.' with '_'
        4. Prepend lora_prefix

    Args:
        model_key: Original model weight key (e.g., "model.diffusion_model.blocks.0.attn.to_q.weight").
        model_prefix: Prefix to strip (e.g., "model.diffusion_model.").
        lora_prefix: LoRA prefix to prepend (e.g., "lora_unet_").

    Returns:
        The LoRA base key name (e.g., "lora_unet_blocks_0_attn_to_q").
    """
    key = model_key
    if model_prefix and key.startswith(model_prefix):
        key = key[len(model_prefix) :]
    # Strip .weight suffix
    if key.endswith(".weight"):
        key = key[: -len(".weight")]
    # Replace . with _
    key = key.replace(".", "_")
    return lora_prefix + key


# --- Filtering ---


def is_extractable_weight(key: str, tensor: torch.Tensor) -> bool:
    """Determine if a weight tensor should have LoRA extracted.

    Returns True for:
    - 2D tensors (Linear layers) with key ending in .weight
    - 4D tensors (Conv2d layers) with key ending in .weight

    Returns False for biases, norms, scalars, embeddings, etc.
    """
    if not key.endswith(".weight"):
        return False
    ndim = tensor.dim()
    return ndim == 2 or ndim == 4


def passes_filter(key: str, include: Optional[list[str]], exclude: Optional[list[str]]) -> bool:
    """Check if a key passes include/exclude substring filters.

    Args:
        key: The model weight key to check.
        include: If not None/empty, key must contain at least one of these substrings.
        exclude: If not None/empty, key must not contain any of these substrings.

    Returns:
        True if the key passes the filter.
    """
    if include:
        if not any(pattern in key for pattern in include):
            return False
    if exclude:
        if any(pattern in key for pattern in exclude):
            return False
    return True


# --- SVD Core ---


def svd_decompose(
    diff: torch.Tensor,
    rank: int,
    clamp_quantile: float,
    use_lowrank: bool = False,
    lowrank_niter: int = 2,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Decompose a difference tensor into LoRA up/down weights via SVD.

    Handles Linear (2D), Conv2d 1x1 (4D with kernel 1x1), and Conv2d 3x3+ (4D).
    S is baked into U (lora_up). Quantile clamping is applied to both U and Vh.

    Args:
        diff: Difference tensor (tuned - original).
        rank: Target LoRA rank.
        clamp_quantile: Quantile for clamping SVD output values (0-1).
        use_lowrank: If True, use torch.svd_lowrank (faster but approximate).
        lowrank_niter: Number of iterations for svd_lowrank.

    Returns:
        (lora_up, lora_down) tensors.
    """
    conv2d = diff.dim() == 4
    kernel_size = None

    if conv2d:
        out_dim, in_dim = diff.shape[:2]
        kernel_size = diff.shape[2:]
        conv2d_3x3 = kernel_size != (1, 1)
        if conv2d_3x3:
            mat = diff.flatten(start_dim=1)  # (out_dim, in_dim * kH * kW)
        else:
            mat = diff.squeeze()  # (out_dim, in_dim)
    else:
        out_dim, in_dim = diff.shape
        mat = diff

    # Rank cannot exceed matrix dimensions
    rank = min(rank, mat.shape[0], mat.shape[1])

    if use_lowrank:
        U, S, V = torch.svd_lowrank(mat, q=rank, niter=lowrank_niter)
        # U: (m, rank), S: (rank,), V: (n, rank)
        Vh = V.t()  # (rank, n)
    else:
        U, S, Vh = torch.linalg.svd(mat, full_matrices=False)
        U = U[:, :rank]
        S = S[:rank]
        Vh = Vh[:rank, :]

    # Bake S into U
    U = U @ torch.diag(S)

    # Quantile clamping
    dist = torch.cat([U.flatten(), Vh.flatten()])
    hi_val = torch.quantile(dist, clamp_quantile)
    low_val = -hi_val
    U = U.clamp(low_val, hi_val)
    Vh = Vh.clamp(low_val, hi_val)

    # Reshape for conv layers
    if conv2d:
        if conv2d_3x3:
            lora_up = U.reshape(out_dim, rank, 1, 1)
            lora_down = Vh.reshape(rank, in_dim, kernel_size[0], kernel_size[1])
        else:
            lora_up = U.reshape(out_dim, rank, 1, 1)
            lora_down = Vh.reshape(rank, in_dim, 1, 1)
    else:
        lora_up = U  # (out_dim, rank)
        lora_down = Vh  # (rank, in_dim)

    return lora_up, lora_down


# --- Per-pair Extraction ---


def extract_from_pair(
    model_org_path: str,
    model_tuned_path: str,
    lora_prefix: str,
    dim: int,
    conv_dim: Optional[int],
    model_key_prefix: Optional[str],
    device: Optional[str],
    save_dtype: Optional[torch.dtype],
    clamp_quantile: float,
    min_diff: float,
    use_lowrank: bool,
    lowrank_niter: int,
    include: Optional[list[str]],
    exclude: Optional[list[str]],
) -> dict[str, torch.Tensor]:
    """Extract LoRA weights from one (original, tuned) model pair.

    Uses MemoryEfficientSafeOpen for lazy per-tensor loading to minimize memory usage.

    Args:
        model_org_path: Path to original model safetensors file.
        model_tuned_path: Path to tuned model safetensors file.
        lora_prefix: LoRA key prefix (e.g., "lora_unet_").
        dim: LoRA rank for Linear layers.
        conv_dim: LoRA rank for Conv2d 3x3 layers. None to skip them.
        model_key_prefix: Prefix to strip from model keys. None for auto-detect.
        device: Device for SVD computation (e.g., "cuda"). None for CPU.
        save_dtype: dtype for output tensors. None for float32.
        clamp_quantile: Quantile for clamping SVD output.
        min_diff: Minimum max-abs difference threshold to extract a layer.
        use_lowrank: Use torch.svd_lowrank.
        lowrank_niter: Iterations for svd_lowrank.
        include: Include filter patterns.
        exclude: Exclude filter patterns.

    Returns:
        State dict fragment with LoRA weights for this model pair.
    """
    if save_dtype is None:
        save_dtype = torch.float32

    logger.info(f"Loading model pair: {model_org_path} / {model_tuned_path}")
    logger.info(f"  LoRA prefix: {lora_prefix}, dim: {dim}, conv_dim: {conv_dim}")

    with safetensors_utils.MemoryEfficientSafeOpen(model_org_path) as org_f, \
         safetensors_utils.MemoryEfficientSafeOpen(model_tuned_path) as tuned_f:

        org_keys = set(org_f.keys())
        tuned_keys = set(tuned_f.keys())
        common_keys = sorted(org_keys & tuned_keys)

        # Warn about asymmetric keys
        only_org = org_keys - tuned_keys
        only_tuned = tuned_keys - org_keys
        if only_org:
            logger.warning(f"  {len(only_org)} key(s) only in original model (skipped)")
        if only_tuned:
            logger.warning(f"  {len(only_tuned)} key(s) only in tuned model (skipped)")

        # Auto-detect or use specified model key prefix
        if model_key_prefix is None:
            detected = detect_model_prefix(list(org_keys), REFERENCE_MODEL_PREFIXES_TO_STRIP)
            if detected:
                logger.info(f"  Auto-detected model key prefix: '{detected}'")
            model_key_prefix = detected
        else:
            logger.info(f"  Using specified model key prefix: '{model_key_prefix}'")

        lora_sd: dict[str, torch.Tensor] = {}
        extracted = 0
        skipped_not_extractable = 0
        skipped_no_diff = 0
        skipped_filtered = 0
        skipped_conv_no_dim = 0

        for key in tqdm(common_keys, desc=f"Extracting ({lora_prefix})"):
            # Load tensors one at a time
            org_tensor = org_f.get_tensor(key)

            if not is_extractable_weight(key, org_tensor):
                skipped_not_extractable += 1
                del org_tensor
                continue

            if not passes_filter(key, include, exclude):
                skipped_filtered += 1
                del org_tensor
                continue

            tuned_tensor = tuned_f.get_tensor(key)

            # Verify shape match
            if org_tensor.shape != tuned_tensor.shape:
                logger.warning(f"  Shape mismatch for {key}: {org_tensor.shape} vs {tuned_tensor.shape}, skipping")
                del org_tensor, tuned_tensor
                continue

            # Compute diff in float32
            diff = tuned_tensor.float() - org_tensor.float()
            del org_tensor, tuned_tensor

            # Check minimum difference
            max_abs_diff = torch.max(torch.abs(diff)).item()
            if max_abs_diff <= min_diff:
                skipped_no_diff += 1
                del diff
                continue

            # Determine if conv and rank
            is_conv2d = diff.dim() == 4
            is_conv2d_3x3 = is_conv2d and diff.shape[2:] != (1, 1)

            if is_conv2d_3x3 and conv_dim is None:
                skipped_conv_no_dim += 1
                del diff
                continue

            rank = conv_dim if is_conv2d_3x3 else dim

            # Move to compute device
            if device:
                diff = diff.to(device)

            # SVD decomposition
            lora_up, lora_down = svd_decompose(diff, rank, clamp_quantile, use_lowrank, lowrank_niter)
            del diff

            # Build LoRA key name
            lora_name = make_lora_key(key, model_key_prefix, lora_prefix)

            # Store results (move to CPU, cast to save_dtype)
            lora_sd[f"{lora_name}.lora_up.weight"] = lora_up.cpu().to(save_dtype).contiguous()
            lora_sd[f"{lora_name}.lora_down.weight"] = lora_down.cpu().to(save_dtype).contiguous()
            lora_sd[f"{lora_name}.alpha"] = torch.tensor(lora_down.shape[0])  # alpha = rank
            del lora_up, lora_down
            extracted += 1

    logger.info(
        f"  Extracted: {extracted}, skipped: {skipped_no_diff} (below min_diff), "
        f"{skipped_not_extractable} (not extractable), {skipped_filtered} (filtered), "
        f"{skipped_conv_no_dim} (conv3x3 without conv_dim)"
    )
    return lora_sd


# --- Orchestration ---


def extract_lora(args: argparse.Namespace) -> None:
    """Main extraction orchestration."""
    # Validate argument lengths
    num_pairs = len(args.model_org)
    if len(args.model_tuned) != num_pairs:
        logger.error(
            f"Number of --model_org ({num_pairs}) and --model_tuned ({len(args.model_tuned)}) must match"
        )
        sys.exit(1)
    if len(args.prefix) != num_pairs:
        logger.error(
            f"Number of --prefix ({len(args.prefix)}) must match --model_org ({num_pairs})"
        )
        sys.exit(1)

    if not args.save_to.lower().endswith(".safetensors"):
        logger.error("--save_to must be a .safetensors file")
        sys.exit(1)

    # Resolve model_key_prefix
    model_key_prefixes: list[Optional[str]]
    if args.model_key_prefix is not None:
        if len(args.model_key_prefix) != num_pairs:
            logger.error(
                f"Number of --model_key_prefix ({len(args.model_key_prefix)}) must match --model_org ({num_pairs})"
            )
            sys.exit(1)
        model_key_prefixes = args.model_key_prefix
    else:
        model_key_prefixes = [None] * num_pairs  # auto-detect for each pair

    save_dtype = str_to_dtype(args.save_precision) if args.save_precision else None

    # Extract from each pair
    combined_sd: dict[str, torch.Tensor] = {}
    for i in range(num_pairs):
        pair_sd = extract_from_pair(
            model_org_path=args.model_org[i],
            model_tuned_path=args.model_tuned[i],
            lora_prefix=args.prefix[i],
            dim=args.dim,
            conv_dim=args.conv_dim,
            model_key_prefix=model_key_prefixes[i],
            device=args.device,
            save_dtype=save_dtype,
            clamp_quantile=args.clamp_quantile,
            min_diff=args.min_diff,
            use_lowrank=args.use_lowrank,
            lowrank_niter=args.lowrank_niter,
            include=args.include,
            exclude=args.exclude,
        )

        # Check for key conflicts
        conflicts = set(pair_sd.keys()) & set(combined_sd.keys())
        if conflicts:
            logger.warning(f"  {len(conflicts)} key conflict(s) between model pairs (overwriting with latest)")

        combined_sd.update(pair_sd)

    if not combined_sd:
        logger.error("No LoRA weights extracted. Check your model pairs and parameters.")
        sys.exit(1)

    logger.info(f"Total LoRA keys: {len(combined_sd)}")

    # Generate metadata
    metadata: dict[str, str] = {}

    net_kwargs: dict[str, str] = {}
    if args.conv_dim is not None:
        net_kwargs["conv_dim"] = str(args.conv_dim)
        net_kwargs["conv_alpha"] = str(float(args.conv_dim))

    metadata[metadata_utils.SS_METADATA_KEY_NETWORK_MODULE] = "networks.lora"
    metadata[metadata_utils.SS_METADATA_KEY_NETWORK_DIM] = str(args.dim)
    metadata[metadata_utils.SS_METADATA_KEY_NETWORK_ALPHA] = str(float(args.dim))
    if net_kwargs:
        metadata[metadata_utils.SS_METADATA_KEY_NETWORK_ARGS] = json.dumps(net_kwargs)

    if not args.no_metadata:
        title = os.path.splitext(os.path.basename(args.save_to))[0]
        metadata_utils.update_title(metadata, title)

    metadata_utils.update_metadata_hashes(combined_sd, metadata)

    # Ensure output directory exists
    dir_name = os.path.dirname(args.save_to)
    if dir_name and not os.path.exists(dir_name):
        os.makedirs(dir_name, exist_ok=True)

    # Save
    logger.info(f"Saving LoRA to: {args.save_to}")
    save_file(combined_sd, args.save_to, metadata=metadata)
    logger.info("Done.")


def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract LoRA from the difference between original and tuned model weights via SVD"
    )

    # Model pair arguments
    parser.add_argument(
        "--model_org",
        type=str,
        nargs="+",
        required=True,
        help="Original model safetensors file(s). Multiple files for multi-component models "
        "(e.g., DiT and Text Encoder separately).",
    )
    parser.add_argument(
        "--model_tuned",
        type=str,
        nargs="+",
        required=True,
        help="Tuned model safetensors file(s). Must match --model_org in count and order.",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        nargs="+",
        required=True,
        help="LoRA key prefix for each model pair (e.g., lora_unet_ lora_te1_). "
        "Must match --model_org in count.",
    )
    parser.add_argument(
        "--save_to",
        type=str,
        required=True,
        help="Output safetensors file path.",
    )

    # LoRA parameters
    parser.add_argument(
        "--dim",
        type=int,
        default=4,
        help="LoRA rank (dimension) for Linear layers (default: 4).",
    )
    parser.add_argument(
        "--conv_dim",
        type=int,
        default=None,
        help="LoRA rank for Conv2d 3x3 layers. If not specified, Conv2d 3x3 layers are skipped.",
    )
    parser.add_argument(
        "--save_precision",
        type=str,
        default=None,
        choices=["float", "fp16", "bf16"],
        help="Precision for saving LoRA weights. Default: float32.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device for SVD computation (e.g., cuda, cuda:0). Default: CPU.",
    )

    # SVD parameters
    parser.add_argument(
        "--clamp_quantile",
        type=float,
        default=0.99,
        help="Quantile for clamping SVD output values (0-1). Default: 0.99.",
    )
    parser.add_argument(
        "--min_diff",
        type=float,
        default=0.01,
        help="Minimum max-abs weight difference to extract a layer. Default: 0.01.",
    )
    parser.add_argument(
        "--use_lowrank",
        action="store_true",
        help="Use torch.svd_lowrank for faster (but approximate) SVD.",
    )
    parser.add_argument(
        "--lowrank_niter",
        type=int,
        default=2,
        help="Number of iterations for torch.svd_lowrank. Default: 2.",
    )

    # Key prefix handling
    parser.add_argument(
        "--model_key_prefix",
        type=str,
        nargs="*",
        default=None,
        help="Prefix to strip from model weight keys (one per model pair). "
        "Use empty string '' for no stripping. "
        "If not specified, auto-detect from known prefixes.",
    )

    # Filtering
    parser.add_argument(
        "--include",
        type=str,
        nargs="*",
        default=None,
        help="Only extract layers whose key contains at least one of these substrings.",
    )
    parser.add_argument(
        "--exclude",
        type=str,
        nargs="*",
        default=None,
        help="Skip layers whose key contains any of these substrings.",
    )

    # Metadata
    parser.add_argument(
        "--no_metadata",
        action="store_true",
        help="Skip modelspec metadata (minimal ss_metadata is always saved).",
    )

    return parser


def main():
    parser = setup_parser()
    add_logging_arguments(parser)
    args = parser.parse_args()
    setup_logging(args, reset=True)
    extract_lora(args)


if __name__ == "__main__":
    main()
