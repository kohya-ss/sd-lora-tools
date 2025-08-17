"""Weight comparison tool for LoRA models."""

import argparse
import sys
import torch

# Relative imports or absolute imports
try:
    from .utils.common import setup_logging, add_logging_arguments
    from .utils import safetensors_utils
except ImportError:
    # If not installed as a package
    import os

    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from sd_lora_tools.utils.common import setup_logging, add_logging_arguments  # type: ignore
    from sd_lora_tools.utils import safetensors_utils  # type: ignore

setup_logging()
import logging

logger = logging.getLogger(__name__)


def load_state_dict(file_name, dtype):
    """Load state dict from file, supporting both safetensors and pytorch formats."""
    if safetensors_utils.is_safetensors(file_name):
        sd, metadata = safetensors_utils.load_safetensors_without_mmap(file_name)
    else:
        sd = torch.load(file_name, map_location="cpu")
        metadata = None

    for key in list(sd.keys()):
        if type(sd[key]) == torch.Tensor:
            sd[key] = sd[key].to(dtype)

    return sd, metadata


def compare_metadatas(file1, file2, metadata1, metadata2):
    keys1 = set(metadata1.keys()) if metadata1 else set()
    keys2 = set(metadata2.keys()) if metadata2 else set()

    print("=== Metadata Comparison Results ===\n")

    # Compare keys
    if keys1 != keys2:
        print("Metadata keys do not match:")
        print(f"  - Only in {file1}: {keys1 - keys2}")
        print(f"  - Only in {file2}: {keys2 - keys1}")

    # Compare values
    for key in keys1 & keys2:
        if metadata1[key] != metadata2[key]:
            print(f"Metadata mismatch for key '{key}':")
            print(f"  - {file1}: {metadata1[key]}")
            print(f"  - {file2}: {metadata2[key]}")

    if keys1 == keys2 and all(metadata1[key] == metadata2[key] for key in keys1):
        print("Metadata matches.")


def compare_weights(file1, file2, rtol, atol, compare_metadata):
    """Compare two weight files and report differences."""
    logger.info(f"Loading first model: {file1}")
    sd1, metadata1 = load_state_dict(file1, torch.float32)

    logger.info(f"Loading second model: {file2}")
    sd2, metadata2 = load_state_dict(file2, torch.float32)

    # Compare metadata
    if compare_metadata:
        compare_metadatas(file1, file2, metadata1, metadata2)

    # Get all keys from both files
    keys1 = set(sd1.keys())
    keys2 = set(sd2.keys())

    # Check for key mismatches
    only_in_file1 = keys1 - keys2
    only_in_file2 = keys2 - keys1
    common_keys = keys1 & keys2

    print("=== Weight File Comparison Results ===\n")

    # Report key mismatches
    if only_in_file1 or only_in_file2:
        print("KEY MISMATCHES DETECTED:")
        if only_in_file1:
            print(f"Keys only in {file1}:")
            for key in sorted(only_in_file1):
                print(f"  - {key}")
        if only_in_file2:
            print(f"Keys only in {file2}:")
            for key in sorted(only_in_file2):
                print(f"  - {key}")
        print()
    else:
        print("✓ All keys match between files\n")

    # Check value mismatches for common keys
    if common_keys:
        print("VALUE COMPARISON:")
        print(f"Tolerance settings: rtol={rtol}, atol={atol}\n")

        mismatched_values = []
        matched_count = 0

        for key in sorted(common_keys):
            tensor1 = sd1[key]
            tensor2 = sd2[key]

            # Check if shapes match
            if tensor1.shape != tensor2.shape:
                print(f"✗ {key}: Shape mismatch - {tensor1.shape} vs {tensor2.shape}")
                mismatched_values.append(key)
                continue

            # Check if values are close
            if torch.allclose(tensor1, tensor2, rtol=rtol, atol=atol):
                matched_count += 1
            else:
                # Calculate detailed difference statistics
                diff = torch.abs(tensor1 - tensor2)
                max_diff = torch.max(diff).item()
                mean_diff = torch.mean(diff).item()

                # Calculate relative error for non-zero values
                non_zero_mask = torch.abs(tensor2) > 1e-10
                if torch.any(non_zero_mask):
                    rel_diff = diff[non_zero_mask] / torch.abs(tensor2[non_zero_mask])
                    max_rel_diff = torch.max(rel_diff).item()
                    mean_rel_diff = torch.mean(rel_diff).item()
                else:
                    max_rel_diff = 0.0
                    mean_rel_diff = 0.0

                print(f"✗ {key}: Value mismatch")
                print(f"    Max absolute diff: {max_diff:.2e}")
                print(f"    Mean absolute diff: {mean_diff:.2e}")
                print(f"    Max relative diff: {max_rel_diff:.2e}")
                print(f"    Mean relative diff: {mean_rel_diff:.2e}")
                print(f"    Shape: {tensor1.shape}")
                mismatched_values.append(key)

        print(f"\nSUMMARY:")
        print(f"Total common keys: {len(common_keys)}")
        print(f"Matching values: {matched_count}")
        print(f"Mismatched values: {len(mismatched_values)}")

        if len(mismatched_values) == 0:
            print("✓ All values match within tolerance")
        else:
            print(f"✗ {len(mismatched_values)} value mismatches detected")

    # Overall result
    print(f"\n=== OVERALL RESULT ===")
    total_mismatches = (
        len(only_in_file1) + len(only_in_file2) + len(mismatched_values) if common_keys else len(only_in_file1) + len(only_in_file2)
    )

    if total_mismatches == 0:
        print("✓ Files are equivalent within specified tolerance")
        return True
    else:
        print(f"✗ Files differ ({total_mismatches} total mismatches)")
        return False


def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare two weight files for equivalence")

    parser.add_argument("file1", type=str, help="First weight file to compare (safetensors or pytorch format)")
    parser.add_argument("file2", type=str, help="Second weight file to compare (safetensors or pytorch format)")
    parser.add_argument("--rtol", type=float, default=1e-5, help="Relative tolerance for torch.allclose (default: 1e-5)")
    parser.add_argument("--atol", type=float, default=1e-8, help="Absolute tolerance for torch.allclose (default: 1e-8)")
    parser.add_argument("--metadata", action="store_true", help="Compare metadata information")

    return parser


def main():
    parser = setup_parser()
    add_logging_arguments(parser)
    args = parser.parse_args()
    setup_logging(args, reset=True)

    result = compare_weights(args.file1, args.file2, args.rtol, args.atol, args.metadata)
    sys.exit(0 if result else 1)


if __name__ == "__main__":
    main()
