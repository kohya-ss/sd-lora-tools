## About LoRA Weight Key Formats

There are multiple conventions for naming keys in LoRA weight files. This document explains those formats.

Except for the sd-scripts / Musubi Tuner format, the information here is based on independent research. If you implement conversion scripts or similar tooling, thorough testing is required.

### Common Rules

LoRA is attached to Linear layers (and sometimes Conv layers) in the base model. In every format, LoRA weight keys are derived from the key name of the corresponding layer weight in the base model, following some naming rules.

For example, if the base layer’s weight key is `blocks.0.self_attn.to_q.weight`, then removing the trailing `weight` yields `blocks.0.self_attn.to_q`, which becomes the source string for constructing the LoRA key. Each format applies its own rules to transform this source string into the “base” part of the LoRA key.

(In LoRA, bias parameters are not trained, so they can usually be ignored.)

A single LoRA weight file may contain LoRA weights for multiple sub-models (for example, U-Net and CLIP), so LoRA keys include a prefix to identify which sub-model they belong to.

For each target layer, the LoRA weights consist of two tensors—LoRA down (LoRA A), which reduces rank, and LoRA up (LoRA B), which increases rank—plus an optional alpha value. The suffix used to distinguish these tensors depends on the format.

In other words, a LoRA key name is constructed as: **prefix + base + suffix**.

### sd-scripts / Musubi Tuner Format

* Base part of the key

  * Replace `.` with `_` in the base model’s weight key name (after removing `.weight`).
  * If the base model key is `blocks.0.self_attn.to_q.weight`, the source is `blocks.0.self_attn.to_q`.

* Prefix

  * For the main image generation model (denoiser), such as U-Net or DiT: `lora_unet_` (even if the implementation is DiT rather than U-Net)
  * For Text Encoders: `lora_te_`. If there are multiple Text Encoders:

    * `lora_te1_` (mainly CLIP-L)
    * `lora_te2_` (mainly CLIP-G)
    * `lora_te3_` (others)
  * Some models may have `lora_te1_` and `lora_te3_` together.

* Suffix

  * `.lora_up.weight`
  * `.lora_down.weight`
  * `.alpha`

So, if the U-Net (DiT) base model key is `blocks.0.self_attn.to_q.weight`, the LoRA keys become:

* `lora_unet_blocks_0_self_attn_to_q.lora_down.weight`
* `lora_unet_blocks_0_self_attn_to_q.lora_up.weight`
* `lora_unet_blocks_0_self_attn_to_q.alpha`

A drawback of this format is that you cannot uniquely reconstruct the original base model key name from the LoRA key. For example, `resblock_up_conv` could correspond to either `resblock.up_conv` or `resblock_up.conv`. When applying LoRA, the base model is known, so this is not a major issue. It becomes problematic when you want to do something purely from the LoRA weights (such as cross-format conversion) without the original model.

### Diffusers and Derived Formats

Note: The main goal of this repository is to support the sd-scripts / Musubi Tuner format and ComfyUI, so Diffusers support is lower priority. We test on ComfyUI, but if weights cannot be loaded in Diffusers, we will address issues as they come in.

* Base part of the key

  * Use the base model’s weight key name as-is (after removing `.weight`).

* Prefix

  * Typically:

    * U-Net: `unet.`
    * Text Encoder(s): `text_encoder.`, `text_encoder_2.`
  * As derivatives (seen in the wild), both U-Net and DiT may use `diffusion_model.` or `transformer.` as the prefix (it is unclear whether these are from official sources or downstream variants).

* Suffix

  * `.lora_A.weight`
  * `.lora_B.weight`
  * `.alpha`

So, if the U-Net (DiT) base model key is `blocks.0.self_attn.to_q.weight`, the LoRA keys may look like:

* `diffusion_model.blocks.0.self_attn.to_q.lora_A.weight`
* `diffusion_model.blocks.0.self_attn.to_q.lora_B.weight`
* `diffusion_model.blocks.0.self_attn.to_q.alpha`

### Formats Supported by ComfyUI

ComfyUI supports multiple derived prefixes and absorbs some variations. As a result, many sd-scripts / Musubi Tuner keys and Diffusers keys work without changes, but additional compatibility handling is still needed.

#### U-Net (DiT)

For some models, only the `diffusion_model.` prefix and Diffusers-style suffixes are supported. Therefore, using the Diffusers format (including the base naming) is generally the safest approach.

#### Text Encoder

For CLIP-L/G and T5XXL, ComfyUI supports both the sd-scripts / Musubi Tuner format and the Diffusers format.
(Models in this category include SD / SDXL / SD3 / FLUX.)

Other Text Encoders use a ComfyUI-specific format.

* The prefix is `text_encoders.`
* The base name is `text_encoders.{state_dict key with .weight removed}`

  * To support multiple Text Encoders, ComfyUI (or its loader) may prepend an identifier prefix such as `qwen3_06b` to the state_dict keys.
* Since the identifier prefixes used by ComfyUI and the key variants of Text Encoder weights (e.g., whether vision components are included) are not well documented, in practice we need to reverse-engineer these per model.
* What is currently known:

  * For Qwen3-0.6B, the model identifier is `qwen3_06b`, and the weights use keys starting with `transformer.model.`
* Using Diffusers-style suffixes seems to be the safest option.

### Supporting LoHa / LoKr

In sd-scripts / Musubi Tuner, the base name and prefix are handled the same way as LoRA. The suffix follows the LyCORIS convention.

Diffusers does not support LoHa/LoKr. As a derived convention, some trainers use Diffusers-style prefix and base name with LyCORIS-style suffixes.

In ComfyUI, the base name and prefix follow the same rules as LoRA in ComfyUI, while the suffix follows the LyCORIS convention. Therefore, to convert from sd-scripts / Musubi Tuner format to a ComfyUI-compatible format, it is recommended to use Diffusers-style base name and prefix, while keeping the suffix unchanged.