# LoRA Extraction

`extract_lora_from_models.py` allows you to extract a LoRA model from the difference between an original base model and a fine-tuned model. This is achieved using SVD (Singular Value Decomposition) to approximate the weights.

> [!IMPORTANT]
> **Dependency Note:** Currently, this script requires an environment where `sd-scripts` is installed to run, as it depends on its internal libraries.

## Basic Usage

```bash
python src/sd_lora_tools/extract_lora_from_models.py 
    --model_org <path_to_original_model> 
    --model_tuned <path_to_tuned_model> 
    --save_to <output_lora_path> 
    --dim 4 
    --device cuda
```

## Options

### Model Settings
- `--model_org`: Path to the original base model (`.ckpt` or `.safetensors`).
- `--model_tuned`: Path to the fine-tuned model.
- `--v2`: Use this if the models are Stable Diffusion v2.x.
- `--sdxl`: Use this if the models are Stable Diffusion XL base.
- `--v_parameterization`: Set metadata for v-parameterization (default follows `--v2`).

### Extraction Parameters
- `--dim`: The rank (dimension) of the extracted LoRA (default: `4`).
- `--conv_dim`: The rank for Conv2d-3x3 layers (default: `None`, disabled).
- `--clamp_quantile`: Quantile clamping value for the difference (default: `0.99`).
- `--min_diff`: Minimum difference threshold to consider for extraction (default: `0.01`).

### Precision and Device
- `--device`: Computation device (`cpu` or `cuda`).
- `--load_precision`: Precision for loading models (`float`, `fp16`, `bf16`).
- `--save_precision`: Precision for saving the LoRA (`float`, `fp16`, `bf16`).

### Miscellaneous
- `--no_metadata`: Do not save `sai` modelspec metadata (minimum `ss_metadata` is still saved).
- `--load_original_model_to` / `--load_tuned_model_to`: Specify memory location (`cpu`, `cuda`) for SDXL models.

---

<details>
<summary>日本語 (Japanese)</summary>

# LoRAの抽出

`extract_lora_from_models.py` は、オリジナルのベースモデルと、それをファインチューニングしたモデルの差分からLoRAモデルを抽出します。これは、SVD（特異値分解）を用いて重みの近似を行うことで実現されています。

> [!IMPORTANT]
> **依存関係に関する注意:** 現在、このスクリプトは内部ライブラリの依存関係により、`sd-scripts` がインストールされた環境での実行が必要です。

## 基本的な使い方

```bash
python src/sd_lora_tools/extract_lora_from_models.py 
    --model_org <オリジナルモデルのパス> 
    --model_tuned <ファインチューニング済モデルのパス> 
    --save_to <出力先LoRAのパス> 
    --dim 4 
    --device cuda
```

## オプション

### モデル設定
- `--model_org`: オリジナルのベースモデルのパス (`.ckpt` または `.safetensors`)。
- `--model_tuned`: ファインチューニング済みモデルのパス。
- `--v2`: Stable Diffusion v2.x系モデルの場合に指定します。
- `--sdxl`: Stable Diffusion XL baseモデルの場合に指定します。
- `--v_parameterization`: v-parameterization用のメタデータを設定します（省略時は `--v2` の設定に従います）。

### 抽出パラメータ
- `--dim`: 抽出されるLoRAのランク（次元数）（デフォルト: `4`）。
- `--conv_dim`: Conv2d-3x3レイヤーのランク（デフォルト: `None`, 無効）。
- `--clamp_quantile`: 差分値のクランプに使用する分位点（デフォルト: `0.99`）。
- `--min_diff`: 抽出対象とする最小の差分のしきい値（デフォルト: `0.01`）。

### 精度とデバイス
- `--device`: 計算デバイス (`cpu` または `cuda`)。
- `--load_precision`: モデル読み込み時の精度 (`float`, `fp16`, `bf16`)。
- `--save_precision`: LoRA保存時の精度 (`float`, `fp16`, `bf16`)。

### その他
- `--no_metadata`: `sai` modelspec メタデータを保存しません（最小限の `ss_metadata` は保存されます）。
- `--load_original_model_to` / `--load_tuned_model_to`: SDXLモデル読み込み時のメモリ配置先 (`cpu`, `cuda`) を指定します。

</details>
