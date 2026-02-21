# LoRA Resizing

`resize.py` is a tool for changing the rank (dimension) of an existing LoRA model. It is primarily used to reduce the size of a high-rank LoRA while preserving as much detail as possible through SVD approximation.

## Basic Usage

```bash
python src/sd_lora_tools/resize.py 
    --model <input_lora.safetensors> 
    --save_to <resized_lora.safetensors> 
    --new_rank 64 
    --device cuda
```

## Options

### Rank Settings
- `--new_rank`: Target rank (dimension) for the output LoRA.
- `--new_conv_rank`: Target rank for Conv2d-3x3 layers (default follows `--new_rank`).

### Dynamic Resizing (Auto rank reduction)
Instead of a fixed rank, you can use dynamic methods to determine the rank based on singular values:
- `--dynamic_method`: Choose a method (`sv_ratio`, `sv_fro`, `sv_cumulative`).
- `--dynamic_param`: Threshold parameter for the chosen dynamic method.
- `--new_rank` acts as a **hard limit** for the maximum rank when using dynamic methods.

### Performance and Precision
- `--device`: Computation device (`cpu` or `cuda`).
- `--save_precision`: Precision for saving the output (`float`, `fp16`, `bf16`).
- `--verbose`: Display detailed information during resizing.

---

<details>
<summary>日本語 (Japanese)</summary>

# LoRAのリサイズ

`resize.py` は、既存のLoRAモデルのランク（次元数）を変更するためのツールです。主に、高ランクのLoRAのサイズを削減しつつ、SVD（特異値分解）による近似を用いて詳細を可能な限り維持するために使用されます。

## 基本的な使い方

```bash
python src/sd_lora_tools/resize.py 
    --model <入力LoRA.safetensors> 
    --save_to <リサイズ後のLoRA.safetensors> 
    --new_rank 64 
    --device cuda
```

## オプション

### ランク設定
- `--new_rank`: 出力LoRAのターゲットランク（次元数）。
- `--new_conv_rank`: Conv2d-3x3レイヤーのターゲットランク（デフォルトは `--new_rank` に従います）。

### ダイナミックリサイズ（自動ランク削減）
固定ランクではなく、特異値に基づいてランクを動的に決定する手法を使用できます。
- `--dynamic_method`: 手法を選択します (`sv_ratio`, `sv_fro`, `sv_cumulative`)。
- `--dynamic_param`: 選択したダイナミックリサイズ手法のしきい値パラメータ。
- ダイナミック手法を使用する場合、 `--new_rank` は最大ランクの **ハードリミット** として機能します。

### パフォーマンスと精度
- `--device`: 計算デバイス (`cpu` または `cuda`)。
- `--save_precision`: 保存時の精度 (`float`, `fp16`, `bf16`)。
- `--verbose`: リサイズプロセス中の詳細な情報を表示します。

</details>
