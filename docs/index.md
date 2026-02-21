# SD LoRA Tools Documentation

This repository provides various utilities for LoRA (Low-Rank Adaptation) models used in Stable Diffusion and other image generation models. These tools are maintained by the same maintainers as the `sd-scripts` repository.

## Table of Contents

- [LoRA Extraction](extract_lora.md) - Extract a LoRA model from the difference between two full models.
- [LoRA Merging](merging.md) - Tools for merging multiple LoRA models or merging LoRA into a base model.
- [LoRA Resizing](resize.md) - Resize a LoRA model to a different rank (dimension).
- [Utilities](utilities.md) - Small tools for checking weights, metadata, and comparing models.

## Installation

```bash
pip install -r requirements.txt
```
*(Note: Ensure you have `torch` and `safetensors` installed in your environment.)*

---

<details>
<summary>日本語 (Japanese)</summary>

このリポジトリは、Stable Diffusionなどの画像生成モデルで使用されるLoRA（Low-Rank Adaptation）モデルのための様々なユーティリティを提供します。これらのツールは `sd-scripts` リポジトリと同じメンテナによってメンテナンスされています。

## 目次

- [LoRAの抽出](extract_lora.md) - 2つのフルモデルの差分からLoRAモデルを抽出します。
- [LoRAのマージ](merging.md) - 複数のLoRAモデルのマージ、またはLoRAをベースモデルにマージするためのツール。
- [LoRAのリサイズ](resize.md) - LoRAモデルを異なるランク（次元数）にリサイズします。
- [ユーティリティ](utilities.md) - 重みの確認、メタデータの表示、モデルの比較を行うためのツール群。

## インストール

```bash
pip install -r requirements.txt
```
*(注: `torch` と `safetensors` が環境にインストールされていることを確認してください。)*

</details>
