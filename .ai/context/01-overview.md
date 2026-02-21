# SD LoRA Tools - Project Context

This repository is a collection of utilities for LoRA (Low-Rank Adaptation) models, maintained by the same team as `sd-scripts`. It focuses on processing LoRA weights for Stable Diffusion and other image generation models.

## Core Scripts (`src/sd_lora_tools/`)

### Extraction & Merging
- **`extract_lora_from_models.py`**: Extracts a LoRA from the difference between an original and a fine-tuned model using SVD.
  - *Note:* Currently depends on `sd-scripts` internal libraries (`library`, `lora`).
- **`svd_merge.py`**: Merges multiple LoRAs into one, supporting rank reduction/SVD approximation.
- **`merge_to_model.py`**: Bakes LoRAs directly into a base checkpoint model.
- **`lora_post_hoc_ema.py`**: Merges weights using Post-Hoc EMA (Exponential Moving Average).

### Processing & Utilities
- **`resize.py`**: Resizes LoRA rank using SVD or dynamic methods (`sv_ratio`, etc.).
- **`check_weights.py`**: Displays weight statistics (min, max, mean).
- **`show_metadata.py`**: Inspects `ss_metadata` within `.safetensors` files.
- **`compare_weights.py`**: Compares two models for weight/metadata equivalence.

## Documentation Structure (`docs/`)
Detailed usage and options are documented in:
- `index.md`: Portal for all tools.
- `extract_lora.md`: Detailed guide for extraction.
- `merging.md`: Guide for merging tools.
- `resize.md`: Guide for resizing and dynamic methods.
- `utilities.md`: Guide for inspection and comparison tools.
- *Format:* English primary text with Japanese translations inside `<details>` tags.

## Engineering Notes
- **Dependencies:** Primarily `torch`, `safetensors`, `numpy`, `tqdm`.
- **Imports:** Most scripts use relative/absolute import fallback logic for both package and direct execution support.
