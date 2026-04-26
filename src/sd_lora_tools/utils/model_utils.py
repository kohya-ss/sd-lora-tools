import sys
from typing import Optional
import torch
from safetensors.torch import save_file

# Relative imports or absolute imports
try:
    from .common import setup_logging
    from . import safetensors_utils
except ImportError:
    # If not installed as a package
    import os

    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from sd_lora_tools.utils.common import setup_logging  # type: ignore
    from sd_lora_tools.utils import safetensors_utils  # type: ignore

setup_logging()
import logging

logger = logging.getLogger(__name__)


def detect_lora_format(lora_keys: list[str]) -> str:
    """Detect LoRA format: 'sd_scripts' (lora_down/lora_up) or 'diffusers' (lora_A/lora_B)."""
    for key in lora_keys:
        if ".lora_down." in key:
            return "sd_scripts"
        if ".lora_A." in key:
            return "diffusers"
    raise ValueError("Could not detect LoRA format from keys (no lora_down or lora_A found)")


def build_model_key_underscore_mapping(
    model_keys: list[str],
    known_prefixes: Optional[list[str]] = None,
) -> dict[str, str]:
    """Build mapping from underscore form (prefix-stripped, dots replaced) to full model key.

    Returns: {underscore_form: full_model_key}
    Example: {"blocks_0_attn_to_q": "model.diffusion_model.blocks.0.attn.to_q.weight"}
    """
    if known_prefixes is None:
        known_prefixes = REFERENCE_MODEL_PREFIXES_TO_STRIP

    mapping: dict[str, str] = {}
    for full_key in model_keys:
        if not full_key.endswith(".weight"):
            continue
        base = full_key[: -len(".weight")]
        for prefix in known_prefixes:
            if base.startswith(prefix):
                stripped = base[len(prefix) :]
                underscore_form = stripped.replace(".", "_")
                if underscore_form not in mapping:
                    mapping[underscore_form] = full_key
                break
    return mapping


def _find_prefix_by_longest_suffix(
    lora_module_name: str,
    valid_underscore_set: dict[str, str],
) -> Optional[tuple[str, str]]:
    """Find the shortest prefix (= longest base match) by scanning '_' positions left to right.

    Returns: (prefix_with_trailing_underscore, full_model_key) or None.
    """
    idx = 0
    while True:
        idx = lora_module_name.find("_", idx)
        if idx == -1:
            break
        candidate_base = lora_module_name[idx + 1 :]
        if candidate_base in valid_underscore_set:
            prefix = lora_module_name[: idx + 1]  # include trailing '_'
            return prefix, valid_underscore_set[candidate_base]
        idx += 1
    return None


def _build_model_base_to_key(
    model_keys: list[str],
    known_prefixes: list[str],
) -> dict[str, str]:
    """Build mapping from dot-form base name (prefix stripped, no .weight) to full model key.

    Returns: {base_dot_form: full_model_key}
    Example: {"blocks.0.attn.to_q": "model.diffusion_model.blocks.0.attn.to_q.weight"}
    """
    mapping: dict[str, str] = {}
    for full_key in model_keys:
        if not full_key.endswith(".weight"):
            continue
        base = full_key[: -len(".weight")]
        for prefix in known_prefixes:
            if base.startswith(prefix):
                stripped = base[len(prefix) :]
                if stripped not in mapping:
                    mapping[stripped] = full_key
                break
    return mapping


def _detect_diffusers_lora_prefix(
    sample_lora_name: str,
    model_base_to_key: dict[str, str],
) -> str:
    """Detect the prefix to strip from diffusers LoRA module names.

    Returns the prefix string (may be empty if direct match works).
    """
    if sample_lora_name in model_base_to_key:
        return ""

    parts = sample_lora_name.split(".")
    for i in range(1, len(parts)):
        suffix = ".".join(parts[i:])
        if suffix in model_base_to_key:
            return ".".join(parts[:i]) + "."

    raise ValueError(f"Could not detect LoRA prefix from sample: {sample_lora_name}")


def _build_sd_scripts_mapping(
    lora_keys: list[str],
    model_keys: list[str],
    known_prefixes: list[str],
) -> dict[str, str]:
    """Build lora_module_name -> full_model_key mapping for sd_scripts format."""
    underscore_mapping = build_model_key_underscore_mapping(model_keys, known_prefixes)

    # Extract unique module names
    module_names: set[str] = set()
    for key in lora_keys:
        if ".lora_down." in key:
            module_name = key.split(".lora_down.")[0]
            module_names.add(module_name)

    mapping: dict[str, str] = {}
    detected_prefixes: set[str] = set()
    unmatched: list[str] = []

    for module_name in sorted(module_names):
        result = _find_prefix_by_longest_suffix(module_name, underscore_mapping)
        if result is not None:
            prefix, model_key = result
            mapping[module_name] = model_key
            detected_prefixes.add(prefix)
        else:
            unmatched.append(module_name)

    if detected_prefixes:
        logger.info(f"Detected sd_scripts LoRA prefixes: {sorted(detected_prefixes)}")
    if unmatched:
        logger.warning(
            f"{len(unmatched)} LoRA modules could not be matched to model keys: "
            f"{unmatched[:5]}{'...' if len(unmatched) > 5 else ''}"
        )

    return mapping


def _build_diffusers_mapping(
    lora_keys: list[str],
    model_keys: list[str],
    known_prefixes: list[str],
) -> dict[str, str]:
    """Build lora_module_name -> full_model_key mapping for diffusers format."""
    model_base_to_key = _build_model_base_to_key(model_keys, known_prefixes)

    # Extract unique module names (dot form)
    module_names: set[str] = set()
    for key in lora_keys:
        if ".lora_A." in key:
            module_name = key.split(".lora_A.")[0]
            module_names.add(module_name)

    if not module_names:
        return {}

    # Detect prefix from a sample
    sample = next(iter(sorted(module_names)))
    lora_prefix = _detect_diffusers_lora_prefix(sample, model_base_to_key)

    if lora_prefix:
        logger.info(f"Detected diffusers LoRA prefix: '{lora_prefix}'")
    else:
        logger.info("Diffusers LoRA: direct key matching (no prefix difference)")

    mapping: dict[str, str] = {}
    unmatched: list[str] = []

    for module_name in sorted(module_names):
        if lora_prefix and module_name.startswith(lora_prefix):
            base = module_name[len(lora_prefix) :]
        else:
            base = module_name

        if base in model_base_to_key:
            mapping[module_name] = model_base_to_key[base]
        else:
            unmatched.append(module_name)

    if unmatched:
        logger.warning(
            f"{len(unmatched)} diffusers LoRA modules could not be matched: "
            f"{unmatched[:5]}{'...' if len(unmatched) > 5 else ''}"
        )

    return mapping


def build_lora_to_model_mapping(
    lora_keys: list[str],
    model_keys: list[str],
    known_prefixes: Optional[list[str]] = None,
) -> tuple[dict[str, str], str]:
    """Build mapping from LoRA module name to full model key.

    Returns: (mapping, format)
    - mapping: {lora_module_name: full_model_key}
    - format: "sd_scripts" or "diffusers"
    """
    if known_prefixes is None:
        known_prefixes = REFERENCE_MODEL_PREFIXES_TO_STRIP

    fmt = detect_lora_format(lora_keys)

    if fmt == "sd_scripts":
        return _build_sd_scripts_mapping(lora_keys, model_keys, known_prefixes), fmt
    else:
        return _build_diffusers_mapping(lora_keys, model_keys, known_prefixes), fmt

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


def dtype_to_str(dtype: torch.dtype) -> str:
    # get name of the dtype
    dtype_name = str(dtype).split(".")[-1]
    return dtype_name


def str_to_dtype(s: Optional[str], default_dtype: Optional[torch.dtype] = None) -> Optional[torch.dtype]:
    """
    Convert a string to a torch.dtype

    Args:
        s: string representation of the dtype
        default_dtype: default dtype to return if s is None

    Returns:
        torch.dtype: the corresponding torch.dtype

    Raises:
        ValueError: if the dtype is not supported

    Examples:
        >>> str_to_dtype(None)
        None
        >>> str_to_dtype(None, torch.float32)
        torch.float32
        >>> str_to_dtype("float32")
        torch.float32
        >>> str_to_dtype("fp32")
        torch.float32
        >>> str_to_dtype("float16")
        torch.float16
        >>> str_to_dtype("fp16")
        torch.float16
        >>> str_to_dtype("bfloat16")
        torch.bfloat16
        >>> str_to_dtype("bf16")
        torch.bfloat16
        >>> str_to_dtype("fp8")
        torch.float8_e4m3fn
        >>> str_to_dtype("fp8_e4m3fn")
        torch.float8_e4m3fn
        >>> str_to_dtype("fp8_e4m3fnuz")
        torch.float8_e4m3fnuz
        >>> str_to_dtype("fp8_e5m2")
        torch.float8_e5m2
        >>> str_to_dtype("fp8_e5m2fnuz")
        torch.float8_e5m2fnuz
    """
    if s is None:
        return default_dtype
    if s in ["bf16", "bfloat16"]:
        return torch.bfloat16
    elif s in ["fp16", "float16"]:
        return torch.float16
    elif s in ["fp32", "float32", "float"]:
        return torch.float32
    elif s in ["fp8_e4m3fn", "e4m3fn", "float8_e4m3fn"]:
        return torch.float8_e4m3fn
    elif s in ["fp8_e4m3fnuz", "e4m3fnuz", "float8_e4m3fnuz"]:
        return torch.float8_e4m3fnuz
    elif s in ["fp8_e5m2", "e5m2", "float8_e5m2"]:
        return torch.float8_e5m2
    elif s in ["fp8_e5m2fnuz", "e5m2fnuz", "float8_e5m2fnuz"]:
        return torch.float8_e5m2fnuz
    elif s in ["fp8", "float8"]:
        return torch.float8_e4m3fn  # default fp8
    else:
        raise ValueError(f"Unsupported dtype: {s}")


class LoRASaverLoader:
    """
    This class handles loading and saving LoRA with transparent conversion from PEFT (Diffusers) to default format.
    """

    DEFAULT_SUFFIX_DOWN = "lora_down"
    DEFAULT_SUFFIX_UP = "lora_up"

    def __init__(self):
        self.format: Optional[str] = None
        self.suffix_down: Optional[str] = None
        self.suffix_up: Optional[str] = None

    @staticmethod
    def _get_format(sd: dict[str, torch.Tensor]) -> Optional[str]:
        for key in list(sd.keys()):
            if "lora_down" in key:  # default
                return "default"
            if "lora_A" in key:  # PEFT LoRA
                return "peft"
        return None

    def load(self, file_name: str, dtype: Optional[torch.dtype]) -> tuple[dict[str, torch.Tensor], Optional[dict[str, str]]]:
        if safetensors_utils.is_safetensors(file_name):
            sd, metadata = safetensors_utils.load_safetensors_without_mmap(file_name, dtype=dtype)
        else:
            sd = torch.load(file_name, map_location="cpu")
            metadata = None

        if self.format is None:
            self.format = self._get_format(sd)
            assert self.format is not None, "Could not determine LoRA format from state_dict"
            if self.format == "default":
                pass  # default LoRA format
            else:
                self.suffix_down = "lora_A"
                self.suffix_up = "lora_B"
        elif self.format != self._get_format(sd):
            raise ValueError(f"LoRA format mismatch. File: {file_name}, Expected: {self.format}, Found: {self._get_format(sd)}")

        for key in list(sd.keys()):
            if type(sd[key]) == torch.Tensor:
                sd[key] = sd[key].to(dtype)

            # force suffix
            if self.suffix_down is not None and self.suffix_down in key:
                new_key = key.replace(self.suffix_down, LoRASaverLoader.DEFAULT_SUFFIX_DOWN)
                sd[new_key] = sd.pop(key)
            elif self.suffix_up is not None and self.suffix_up in key:
                new_key = key.replace(self.suffix_up, LoRASaverLoader.DEFAULT_SUFFIX_UP)
                sd[new_key] = sd.pop(key)

        return sd, metadata  # type: ignore

    def save(self, file_name: str, state_dict: dict[str, torch.Tensor], metadata: Optional[dict[str, str]]):
        # restore suffix
        if self.suffix_down is not None or self.suffix_up is not None:
            for key in list(state_dict.keys()):
                if self.suffix_down is not None and LoRASaverLoader.DEFAULT_SUFFIX_DOWN in key:
                    new_key = key.replace(LoRASaverLoader.DEFAULT_SUFFIX_DOWN, self.suffix_down)
                    state_dict[new_key] = state_dict.pop(key)
                elif self.suffix_up is not None and LoRASaverLoader.DEFAULT_SUFFIX_UP in key:
                    new_key = key.replace(LoRASaverLoader.DEFAULT_SUFFIX_UP, self.suffix_up)
                    state_dict[new_key] = state_dict.pop(key)

        if safetensors_utils.is_safetensors(file_name):
            save_file(state_dict, file_name, metadata)
        else:
            torch.save(state_dict, file_name)
