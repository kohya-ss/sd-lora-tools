# LoRA Extraction

`extract_lora.py` extracts a LoRA model from the difference between original and tuned model weights using SVD (Singular Value Decomposition). It operates directly on safetensors weight files without model instantiation, making it architecture-agnostic — it works with SD, SDXL, FLUX, SD3, HunyuanVideo, and any other model that stores weights in safetensors format.

The output is in **sd-scripts format**. To convert to Diffusers format, use [`convert_lora.py`](convert_lora.md).

## Basic Usage

Extract LoRA from a single model pair (e.g., a DiT model):

```bash
python -m sd_lora_tools.extract_lora \
    --model_org base_dit.safetensors \
    --model_tuned tuned_dit.safetensors \
    --prefix lora_unet_ \
    --save_to output.safetensors \
    --dim 16 \
    --device cuda
```

## Multi-Component Models

For models with separate weight files (e.g., DiT + Text Encoder), specify multiple pairs:

```bash
python -m sd_lora_tools.extract_lora \
    --model_org dit.safetensors text_encoder.safetensors \
    --model_tuned dit_tuned.safetensors text_encoder_tuned.safetensors \
    --prefix lora_unet_ lora_te1_ \
    --save_to output.safetensors \
    --dim 16 \
    --device cuda
```

Each `--model_org`, `--model_tuned`, and `--prefix` entry corresponds by position. All three arguments must have the same number of values.

## Options

### Model Pair Arguments
- `--model_org`: Original model safetensors file(s).
- `--model_tuned`: Tuned model safetensors file(s). Must match `--model_org` in count.
- `--prefix`: LoRA key prefix for each pair (e.g., `lora_unet_`, `lora_te1_`, `lora_te2_`). Must match `--model_org` in count.
- `--save_to`: Output safetensors file path.

### LoRA Rank
- `--dim`: Rank for Linear layers (default: `4`).
- `--conv_dim`: Rank for Conv2d 3x3 layers (default: `None` — Conv2d 3x3 layers are skipped).

### SVD Parameters
- `--clamp_quantile`: Quantile clamping value (default: `0.99`).
- `--min_diff`: Minimum max-abs difference to extract a layer (default: `0.01`).
- `--use_lowrank`: Use `torch.svd_lowrank` for faster approximate SVD.
- `--lowrank_niter`: Number of iterations for `svd_lowrank` (default: `2`).

### Precision and Device
- `--device`: Device for SVD computation (`cuda`, `cuda:0`, etc.). Default: CPU.
- `--save_precision`: Precision for saving (`float`, `fp16`, `bf16`). Default: float32.

### Key Prefix Handling
- `--model_key_prefix`: Prefix to strip from model weight keys, one per model pair. Use empty string `""` for no stripping. If not specified, auto-detection is used (tries known prefixes like `model.diffusion_model.`, `model.`, etc.).

### Layer Filtering
- `--include`: Only extract layers whose key contains at least one of these substrings (e.g., `--include attn proj`).
- `--exclude`: Skip layers whose key contains any of these substrings (e.g., `--exclude time_embed norm`).

### Metadata
- `--no_metadata`: Skip modelspec metadata (minimal ss_metadata for LoRA is always saved).

## Examples

### Fast Extraction with Approximate SVD

```bash
python -m sd_lora_tools.extract_lora \
    --model_org base.safetensors \
    --model_tuned tuned.safetensors \
    --prefix lora_unet_ \
    --save_to output.safetensors \
    --dim 32 \
    --use_lowrank \
    --device cuda
```

### Extract Only Attention Layers

```bash
python -m sd_lora_tools.extract_lora \
    --model_org base.safetensors \
    --model_tuned tuned.safetensors \
    --prefix lora_unet_ \
    --save_to output.safetensors \
    --dim 16 \
    --include attn
```

### With Explicit Model Key Prefix

```bash
python -m sd_lora_tools.extract_lora \
    --model_org dit.safetensors te.safetensors \
    --model_tuned dit_tuned.safetensors te_tuned.safetensors \
    --prefix lora_unet_ lora_te1_ \
    --model_key_prefix "model.diffusion_model." "" \
    --save_to output.safetensors \
    --dim 16
```

---

<details>
<summary>日本語 (Japanese)</summary>

# LoRAの抽出

`extract_lora.py` は、元モデルと派生（ファインチューニング済み）モデルの重みの差分から、SVD（特異値分解）を用いてLoRAモデルを抽出します。safetensorsの重みファイルを直接操作し、モデルのインスタンス化を必要としないため、アーキテクチャに依存しません。SD、SDXL、FLUX、SD3、HunyuanVideoなど、safetensors形式で重みを保存するあらゆるモデルに対応します。

出力は **sd-scripts形式** です。Diffusers形式に変換するには [`convert_lora.py`](convert_lora.md) を使用してください。

## 基本的な使い方

単一のモデルペアからLoRAを抽出する場合（例: DiTモデル）：

```bash
python -m sd_lora_tools.extract_lora \
    --model_org base_dit.safetensors \
    --model_tuned tuned_dit.safetensors \
    --prefix lora_unet_ \
    --save_to output.safetensors \
    --dim 16 \
    --device cuda
```

## 複数コンポーネントのモデル

重みファイルが分かれているモデル（例: DiT + Text Encoder）の場合、複数のペアを指定します：

```bash
python -m sd_lora_tools.extract_lora \
    --model_org dit.safetensors text_encoder.safetensors \
    --model_tuned dit_tuned.safetensors text_encoder_tuned.safetensors \
    --prefix lora_unet_ lora_te1_ \
    --save_to output.safetensors \
    --dim 16 \
    --device cuda
```

`--model_org`、`--model_tuned`、`--prefix` の各エントリは位置で対応します。3つの引数すべて同じ数の値が必要です。

## オプション

### モデルペア引数
- `--model_org`: 元モデルのsafetensorsファイル。複数指定可能。
- `--model_tuned`: 派生モデルのsafetensorsファイル。`--model_org` と同数必要。
- `--prefix`: 各ペアのLoRAキーprefix（例: `lora_unet_`、`lora_te1_`、`lora_te2_`）。`--model_org` と同数必要。
- `--save_to`: 出力safetensorsファイルパス。

### LoRAランク
- `--dim`: Linear層のランク（デフォルト: `4`）。
- `--conv_dim`: Conv2d 3x3層のランク（デフォルト: `None` — Conv2d 3x3層はスキップ）。

### SVDパラメータ
- `--clamp_quantile`: クランプ分位値（デフォルト: `0.99`）。
- `--min_diff`: レイヤーを抽出するための最小最大絶対差分（デフォルト: `0.01`）。
- `--use_lowrank`: 高速な近似SVDとして `torch.svd_lowrank` を使用。
- `--lowrank_niter`: `svd_lowrank` の反復回数（デフォルト: `2`）。

### 精度とデバイス
- `--device`: SVD計算デバイス（`cuda`、`cuda:0` 等）。デフォルト: CPU。
- `--save_precision`: 保存精度（`float`、`fp16`、`bf16`）。デフォルト: float32。

### キーprefix処理
- `--model_key_prefix`: モデル重みキーから除去するprefix（モデルペアごとに1つ）。除去しない場合は空文字列 `""` を使用。未指定の場合、既知のprefix（`model.diffusion_model.`、`model.` 等）から自動検出。

### レイヤーフィルタリング
- `--include`: 指定サブストリングのいずれかを含むレイヤーのみ抽出（例: `--include attn proj`）。
- `--exclude`: 指定サブストリングのいずれかを含むレイヤーをスキップ（例: `--exclude time_embed norm`）。

### メタデータ
- `--no_metadata`: modelspecメタデータを省略（LoRAの最低限のss_metadataは常に保存されます）。

## 使用例

### 近似SVDによる高速抽出

```bash
python -m sd_lora_tools.extract_lora \
    --model_org base.safetensors \
    --model_tuned tuned.safetensors \
    --prefix lora_unet_ \
    --save_to output.safetensors \
    --dim 32 \
    --use_lowrank \
    --device cuda
```

### Attention層のみ抽出

```bash
python -m sd_lora_tools.extract_lora \
    --model_org base.safetensors \
    --model_tuned tuned.safetensors \
    --prefix lora_unet_ \
    --save_to output.safetensors \
    --dim 16 \
    --include attn
```

### モデルキーprefixの明示指定

```bash
python -m sd_lora_tools.extract_lora \
    --model_org dit.safetensors te.safetensors \
    --model_tuned dit_tuned.safetensors te_tuned.safetensors \
    --prefix lora_unet_ lora_te1_ \
    --model_key_prefix "model.diffusion_model." "" \
    --save_to output.safetensors \
    --dim 16
```

</details>
