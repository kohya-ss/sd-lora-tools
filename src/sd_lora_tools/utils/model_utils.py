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
