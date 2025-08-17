import hashlib
from io import BytesIO
import json
import os
from typing import Optional
import safetensors.torch
import torch


# this metadata is referred from train_network and various scripts, so we wrote here
SS_METADATA_KEY_V2 = "ss_v2"
SS_METADATA_KEY_BASE_MODEL_VERSION = "ss_base_model_version"
SS_METADATA_KEY_NETWORK_MODULE = "ss_network_module"
SS_METADATA_KEY_NETWORK_DIM = "ss_network_dim"
SS_METADATA_KEY_NETWORK_ALPHA = "ss_network_alpha"
SS_METADATA_KEY_NETWORK_ARGS = "ss_network_args"

SS_METADATA_MINIMUM_KEYS = [
    SS_METADATA_KEY_V2,
    SS_METADATA_KEY_BASE_MODEL_VERSION,
    SS_METADATA_KEY_NETWORK_MODULE,
    SS_METADATA_KEY_NETWORK_DIM,
    SS_METADATA_KEY_NETWORK_ALPHA,
    SS_METADATA_KEY_NETWORK_ARGS,
]

MODELSPEC_TITLE = "modelspec.title"
MODELSPEC_MERGED_FROM = "modelspec.merged_from"


def update_metadata_dim_alpha(
    metadata: dict[str, str], network_dim: str, network_alpha: str, network_args: Optional[dict[str, str]] = None
):
    if SS_METADATA_KEY_NETWORK_DIM in metadata:
        metadata[SS_METADATA_KEY_NETWORK_DIM] = network_dim
    if SS_METADATA_KEY_NETWORK_ALPHA in metadata:
        metadata[SS_METADATA_KEY_NETWORK_ALPHA] = network_alpha
    if network_args is not None:
        metadata[SS_METADATA_KEY_NETWORK_ARGS] = json.dumps(network_args)
    return metadata


def update_title(metadata: dict[str, str], title: str):
    metadata[MODELSPEC_TITLE] = title
    return metadata


def update_metadata_hashes(state_dict: dict[str, torch.Tensor], metadata: dict[str, str]):
    model_hash, legacy_hash = precalculate_safetensors_hashes(state_dict, metadata)
    metadata["sshs_model_hash"] = model_hash
    metadata["sshs_legacy_hash"] = legacy_hash


def precalculate_safetensors_hashes(tensors, metadata):
    """Precalculate the model hashes needed by sd-webui-additional-networks to
    save time on indexing the model later."""

    # Because writing user metadata to the file can change the result of
    # sd_models.model_hash(), only retain the training metadata for purposes of
    # calculating the hash, as they are meant to be immutable
    metadata = {k: v for k, v in metadata.items() if k.startswith("ss_")}

    bytes = safetensors.torch.save(tensors, metadata)
    b = BytesIO(bytes)

    model_hash = addnet_hash_safetensors(b)
    legacy_hash = addnet_hash_legacy(b)
    return model_hash, legacy_hash


def addnet_hash_legacy(b):
    """Old model hash used by sd-webui-additional-networks for .safetensors format files"""
    m = hashlib.sha256()

    b.seek(0x100000)
    m.update(b.read(0x10000))
    return m.hexdigest()[0:8]


def addnet_hash_safetensors(b):
    """New model hash used by sd-webui-additional-networks for .safetensors format files"""
    hash_sha256 = hashlib.sha256()
    blksize = 1024 * 1024

    b.seek(0)
    header = b.read(8)
    n = int.from_bytes(header, "little")

    offset = n + 8
    b.seek(offset)
    for chunk in iter(lambda: b.read(blksize), b""):
        hash_sha256.update(chunk)

    return hash_sha256.hexdigest()


def update_merged_from(metadata: dict[str, str], metadatas: list[Optional[dict[str, str]]], model_names: list[str]):
    """Update the metadata with the merged_from field."""
    model_titles = []
    for i, (metadata_i, model_name) in enumerate(zip(metadatas, model_names)):
        if metadata_i is not None and MODELSPEC_TITLE in metadata_i:
            title = metadata_i[MODELSPEC_TITLE]
            if MODELSPEC_MERGED_FROM in metadata_i:
                title = f"{title} ({metadata_i[MODELSPEC_MERGED_FROM]})"
            model_titles.append(title)
        else:
            model_titles.append(os.path.splitext(os.path.basename(model_name))[0])

    if model_titles:
        metadata[MODELSPEC_MERGED_FROM] = ", ".join(model_titles)
    return metadata
