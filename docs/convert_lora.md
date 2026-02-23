# LoRA Format Conversion

Converts LoRA weight files between **sd-scripts / Musubi Tuner format** and **Diffusers format**. This is a generic, model-agnostic converter that uses a reference model's key names to resolve naming ambiguities.

## Overview

sd-scripts format and Diffusers format differ in three ways:

| | sd-scripts | Diffusers |
|---|---|---|
| **Key base** | `.` replaced with `_` | Original model key names |
| **Suffix** | `.lora_down.weight` / `.lora_up.weight` | `.lora_A.weight` / `.lora_B.weight` |
| **Alpha** | Stored as `.alpha` key | Not stored (baked into weights) |

Because `_` to `.` conversion is ambiguous (e.g., `blocks_0_self_attn` could be `blocks.0.self_attn` or `blocks.0.self.attn`), converting from sd-scripts to Diffusers requires a **reference model** to resolve the correct key names.

## Usage

### sd-scripts → Diffusers

```bash
python -m sd_lora_tools.convert_lora \
  --input lora_sd_scripts.safetensors \
  --output lora_diffusers.safetensors \
  --target diffusers \
  --reference_model base_model.safetensors
```

If the UNet and Text Encoder weights are in separate files, specify multiple reference models:

```bash
python -m sd_lora_tools.convert_lora \
  --input lora_sd_scripts.safetensors \
  --output lora_diffusers.safetensors \
  --target diffusers \
  --reference_model unet.safetensors text_encoder.safetensors
```

### Diffusers → sd-scripts

No reference model is needed for this direction (`.` → `_` is unambiguous).

```bash
python -m sd_lora_tools.convert_lora \
  --input lora_diffusers.safetensors \
  --output lora_sd_scripts.safetensors \
  --target sd_scripts
```

## Options

### Required Arguments

| Argument | Description |
|---|---|
| `--input` | Input LoRA safetensors file |
| `--output` | Output safetensors file |
| `--target` | Target format: `sd_scripts` or `diffusers` |
| `--reference_model` | Reference model file(s) for key name resolution (required for `--target diffusers` only). Multiple files can be specified. |

### Prefix Overrides

By default, prefixes are auto-detected from the input keys. You can override them:

| Argument | Default |
|---|---|
| `--sd_scripts_unet_prefix` | `lora_unet_` |
| `--sd_scripts_te_prefix` | Auto: `lora_te_` (single TE) or `lora_te1_`, `lora_te2_`, ... (multiple TEs) |
| `--diffusers_unet_prefix` | `diffusion_model.` |
| `--diffusers_te_prefix` | Auto: `text_encoder.` (single TE) or `text_encoder.`, `text_encoder_2.`, ... (multiple TEs) |

### Other Options

| Argument | Description |
|---|---|
| `--save_precision` | Output precision: `float`, `fp16`, or `bf16` (default: same as input) |

## Alpha Handling

### sd-scripts → Diffusers

Alpha values are baked into the weights and the `.alpha` keys are removed. The scaling factor `alpha / rank` is split between `lora_down` and `lora_up` using a power-of-2 decomposition to minimize floating-point rounding error:

- The scale is decomposed as `2^k * (odd/odd)`
- The `2^k` part is split evenly between both matrices (exact, no rounding)
- The remaining odd fraction is applied to one matrix only (at most 1 rounding operation)

For example, with `rank=32, alpha=1`: scale = 1/32 = 1/8 × 1/4 (both powers of 2, **zero rounding error**).

When `alpha == rank` (the most common case), scale = 1 and no modification is needed.

### Diffusers → sd-scripts

Alpha is set to `rank` for each module, so the effective scale is 1. No weight modification is needed.

## Text Encoder Support

The script supports multiple Text Encoders. Prefix detection is automatic:

**sd-scripts format:**
- Single TE: `lora_te_`
- Multiple TEs: `lora_te1_`, `lora_te2_`, ...

**Diffusers format:**
- Single TE: `text_encoder.`
- Multiple TEs: `text_encoder.`, `text_encoder_2.`, ...

Use the `--sd_scripts_te_prefix` and `--diffusers_te_prefix` options to override if the auto-detection does not match your model.

---

<details>
<summary>日本語 (Japanese)</summary>

# LoRAフォーマット変換

LoRA重みファイルを **sd-scripts / Musubi Tuner形式** と **Diffusers形式** の間で変換します。参照モデルのキー名を利用する汎用的なコンバーターで、特定のモデルアーキテクチャに依存しません。

## 概要

sd-scripts形式とDiffusers形式には3つの違いがあります：

| | sd-scripts | Diffusers |
|---|---|---|
| **キーのベース名** | `.` を `_` に置換 | 元モデルのキー名そのまま |
| **サフィックス** | `.lora_down.weight` / `.lora_up.weight` | `.lora_A.weight` / `.lora_B.weight` |
| **Alpha** | `.alpha` キーとして保存 | 保存しない（重みにbake） |

`_` から `.` への逆変換は曖昧なため（例: `blocks_0_self_attn` は `blocks.0.self_attn` にも `blocks.0.self.attn` にもなり得る）、sd-scriptsからDiffusersへの変換には正しいキー名を解決するための**参照モデル**が必要です。

## 使い方

### sd-scripts → Diffusers

```bash
python -m sd_lora_tools.convert_lora \
  --input lora_sd_scripts.safetensors \
  --output lora_diffusers.safetensors \
  --target diffusers \
  --reference_model base_model.safetensors
```

UNetとText Encoderの重みが別ファイルの場合、複数の参照モデルを指定できます：

```bash
python -m sd_lora_tools.convert_lora \
  --input lora_sd_scripts.safetensors \
  --output lora_diffusers.safetensors \
  --target diffusers \
  --reference_model unet.safetensors text_encoder.safetensors
```

### Diffusers → sd-scripts

この方向では参照モデルは不要です（`.` → `_` の変換は一意です）。

```bash
python -m sd_lora_tools.convert_lora \
  --input lora_diffusers.safetensors \
  --output lora_sd_scripts.safetensors \
  --target sd_scripts
```

## オプション

### 必須引数

| 引数 | 説明 |
|---|---|
| `--input` | 入力LoRA safetensorsファイル |
| `--output` | 出力safetensorsファイル |
| `--target` | 変換先フォーマット: `sd_scripts` または `diffusers` |
| `--reference_model` | キー名解決用の参照モデルファイル（`--target diffusers` の場合のみ必須）。複数ファイル指定可。 |

### プレフィックスの上書き

デフォルトでは入力キーからプレフィックスを自動検出します。手動で上書きすることも可能です：

| 引数 | デフォルト |
|---|---|
| `--sd_scripts_unet_prefix` | `lora_unet_` |
| `--sd_scripts_te_prefix` | 自動: `lora_te_`（単一TE）または `lora_te1_`, `lora_te2_`, ...（複数TE） |
| `--diffusers_unet_prefix` | `diffusion_model.` |
| `--diffusers_te_prefix` | 自動: `text_encoder.`（単一TE）または `text_encoder.`, `text_encoder_2.`, ...（複数TE） |

### その他のオプション

| 引数 | 説明 |
|---|---|
| `--save_precision` | 出力精度: `float`, `fp16`, `bf16`（デフォルト: 入力と同じ） |

## Alphaの扱い

### sd-scripts → Diffusers

Alpha値を重みにbakeし、`.alpha` キーは削除します。スケーリング係数 `alpha / rank` は、浮動小数点の丸め誤差を最小化するため、2の冪乗分解を用いて `lora_down` と `lora_up` に分配されます：

- スケールを `2^k × (奇数/奇数)` に分解
- `2^k` 部分を両方の行列に均等分配（正確、丸め誤差なし）
- 残りの奇数部分は片方の行列にのみ適用（丸め操作は最大1回）

例: `rank=32, alpha=1` の場合、scale = 1/32 = 1/8 × 1/4（両方2の冪乗、**丸め誤差ゼロ**）。

`alpha == rank`（最も一般的なケース）の場合、scale = 1 なので重みの変更は不要です。

### Diffusers → sd-scripts

各モジュールの alpha を `rank` と同じ値に設定するため、実効スケールは1です。重みの変更は不要です。

## Text Encoderサポート

複数のText Encoderをサポートしています。プレフィックスの検出は自動です：

**sd-scripts形式：**
- 単一TE: `lora_te_`
- 複数TE: `lora_te1_`, `lora_te2_`, ...

**Diffusers形式：**
- 単一TE: `text_encoder.`
- 複数TE: `text_encoder.`, `text_encoder_2.`, ...

自動検出がモデルに合わない場合は、`--sd_scripts_te_prefix` と `--diffusers_te_prefix` オプションで上書きしてください。

</details>
