# LoRA Merging

This section covers tools for merging multiple LoRAs together or merging them into a base model.

## 1. SVD Merge (`svd_merge.py`)

This tool merges multiple LoRA models into a single LoRA model. It can use SVD (Singular Value Decomposition) to approximate the merged weights, allowing for rank reduction.

### Basic Usage

```bash
python src/sd_lora_tools/svd_merge.py 
    --models <model1.safetensors> <model2.safetensors> 
    --ratios 0.5 0.5 
    --save_to <merged_lora.safetensors> 
    --new_rank 64 
    --device cuda
```

### Options

- `--models`: List of LoRA models to merge.
- `--ratios`: Ratio for each model in the same order.
- `--new_rank`: Target rank for the merged LoRA.
- `--new_conv_rank`: Target rank for Conv2d layers.
- `--regex_scales`: Apply specific scales to specific layers using regex (e.g., LBW).
- `--use_svd_lowrank`: Use `torch.svd_lowrank` for the merging process.

## 2. Merge to Model (`merge_to_model.py`)

This tool "bakes" one or more LoRAs directly into a base checkpoint model.

### Basic Usage

```bash
python src/sd_lora_tools/merge_to_model.py 
    --base_model <base_model.safetensors> 
    --models <lora1.safetensors> <lora2.safetensors> 
    --ratios 1.0 0.5 
    --save_to <output_model.safetensors> 
    --device cuda
```

### Options

- `--base_model`: The original checkpoint to merge into.
- `--models`: One or more LoRA files to merge.
- `--ratios`: The weight (strength) for each LoRA.
- `--regex_scales`: Per-layer scaling using regular expressions.

## 3. Post-Hoc EMA Merge (`lora_post_hoc_ema.py`)

Merges multiple LoRA weights using the Post-Hoc EMA (Exponential Moving Average) method. This is typically used for merging a series of checkpoints from the same training run.

### Basic Usage

```bash
python src/sd_lora_tools/lora_post_hoc_ema.py 
    <lora_step_100.safetensors> <lora_step_200.safetensors> ... 
    --output_file <ema_merged.safetensors> 
    --beta 0.999
```

### Options

- `--beta`: Decay rate for merging weights.
- `--beta2`: Decay rate for linear interpolation.
- `--sigma_rel`: Relative sigma for Power Function EMA.
- `--no_sort`: Do not sort input files by modification time.

---

<details>
<summary>日本語 (Japanese)</summary>

# LoRAのマージ

このセクションでは、複数のLoRAを一つにまとめたり、ベースモデルに適用（焼き込み）したりするためのツールについて説明します。

## 1. SVD Merge (`svd_merge.py`)

複数のLoRAモデルを一つのLoRAモデルにマージします。SVD（特異値分解）を用いてマージ後の重みを近似することで、ランクの削減（リサイズ）を同時に行うことが可能です。

### 基本的な使い方

```bash
python src/sd_lora_tools/svd_merge.py 
    --models <model1.safetensors> <model2.safetensors> 
    --ratios 0.5 0.5 
    --save_to <merged_lora.safetensors> 
    --new_rank 64 
    --device cuda
```

### 主なオプション

- `--models`: マージするLoRAモデルのリスト。
- `--ratios`: 各モデルに対応するマージ比率。
- `--new_rank`: 出力されるLoRAのランク。
- `--new_conv_rank`: Conv2dレイヤーの出力ランク。
- `--regex_scales`: 正規表現を用いて特定のレイヤー（LBWなど）に異なる比率を適用します。
- `--use_svd_lowrank`: 計算に `torch.svd_lowrank` を使用します。

## 2. Merge to Model (`merge_to_model.py`)

一つまたは複数のLoRAを、ベースとなるチェックポイントモデルに直接マージして保存します。

### 基本的な使い方

```bash
python src/sd_lora_tools/merge_to_model.py 
    --base_model <ベースモデル.safetensors> 
    --models <lora1.safetensors> <lora2.safetensors> 
    --ratios 1.0 0.5 
    --save_to <出力モデル.safetensors> 
    --device cuda
```

### 主なオプション

- `--base_model`: マージ先となるオリジナルのチェックポイント。
- `--models`: マージするLoRAファイルのリスト。
- `--ratios`: 各LoRAの適用強度。
- `--regex_scales`: 正規表現によるレイヤーごとのスケーリング指定。

## 3. Post-Hoc EMA Merge (`lora_post_hoc_ema.py`)

Post-Hoc EMA（指数移動平均）法を用いて複数のLoRAの重みをマージします。通常、同じ学習ランから得られた一連のチェックポイントを統合するために使用されます。

### 基本的な使い方

```bash
python src/sd_lora_tools/lora_post_hoc_ema.py 
    <lora_step_100.safetensors> <lora_step_200.safetensors> ... 
    --output_file <ema_merged.safetensors> 
    --beta 0.999
```

### 主なオプション

- `--beta`: 重みマージの減衰率。
- `--beta2`: 線形補間のための減衰率。
- `--sigma_rel`: Power Function EMA 用の相対的なシグマ値。
- `--no_sort`: 入力ファイルを更新日時でソートしないようにします。

</details>
