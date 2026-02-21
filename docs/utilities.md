# Utilities

This collection of small tools helps with checking, inspecting, and comparing LoRA weight files.

## 1. Check Weights (`check_weights.py`)

A simple utility to check the statistics of weight tensors in a model file.

### Usage
```bash
python src/sd_lora_tools/check_weights.py <model.safetensors>
```
- `-s, --show_all_keys`: Display all tensor keys in the model.

## 2. Show Metadata (`show_metadata.py`)

Displays the metadata stored within a `.safetensors` model file (e.g., training settings, base model information).

### Usage
```bash
python src/sd_lora_tools/show_metadata.py --model <model.safetensors>
```

## 3. Compare Weights (`compare_weights.py`)

Compares two weight files to check if they are identical or within a certain tolerance.

### Usage
```bash
python src/sd_lora_tools/compare_weights.py <file1.safetensors> <file2.safetensors>
```

### Options
- `--rtol`: Relative tolerance (default: `1e-5`).
- `--atol`: Absolute tolerance (default: `1e-8`).
- `--metadata`: Compare metadata fields in addition to weights.

---

<details>
<summary>日本語 (Japanese)</summary>

# ユーティリティ

LoRAの重みファイルのチェック、検査、比較を行うための小規模なツール群です。

## 1. Check Weights (`check_weights.py`)

モデルファイル内の重みテンソルの統計情報を確認するためのシンプルなユーティリティです。

### 使い方
```bash
python src/sd_lora_tools/check_weights.py <モデル.safetensors>
```
- `-s, --show_all_keys`: モデル内のすべてのテンソルキーを表示します。

## 2. Show Metadata (`show_metadata.py`)

`.safetensors` モデルファイル内に保存されているメタデータ（学習設定、ベースモデル情報など）を表示します。

### 使い方
```bash
python src/sd_lora_tools/show_metadata.py --model <モデル.safetensors>
```

## 3. Compare Weights (`compare_weights.py`)

2つの重みファイルが同一であるか、あるいは特定の許容範囲内であるかを比較します。

### 使い方
```bash
python src/sd_lora_tools/compare_weights.py <ファイル1.safetensors> <ファイル2.safetensors>
```

### 主なオプション
- `--rtol`: 相対許容誤差 (デフォルト: `1e-5`)。
- `--atol`: 絶対許容誤差 (デフォルト: `1e-8`)。
- `--metadata`: 重みに加えてメタデータフィールドも比較します。

</details>
