# Lab 1 — Fine-Tune with LoRA

**Book 3 · Part A, Chapter 4 (LoRA, the training pipeline)**

---

## Hypothesis

LoRA adapts a pretrained model for a new task using a fraction of full fine-tuning's memory, because only the small low-rank adapter matrices carry gradients and optimizer state. The base model's weights are frozen throughout.

## Background

LoRA injects two small matrices A (d×r) and B (r×d) alongside each frozen weight matrix W (d×d). The effective weight update is ΔW = BA, where r ≪ d. The number of trainable parameters per layer is `2 × r × d` instead of `d²`. For a 7B model with d=4096 and r=8, this is roughly 0.07% of the total parameters. Only the adapter matrices accumulate gradients, so optimizer state (Adam momentum + variance) also shrinks proportionally. This is why LoRA fits where full fine-tuning OOMs.

## Setup

```bash
pip install peft transformers datasets trl accelerate bitsandbytes

# A base model for instruction fine-tuning
# Qwen2.5-3B-Instruct or Llama-3.2-3B work well on a single consumer GPU
```

## Procedure

### Part A — LoRA trainable parameter count
1. Load a base model with `AutoModelForCausalLM.from_pretrained(...)`.
2. Apply a LoRA config and call `get_peft_model(model, config)`:
   ```python
   from peft import LoraConfig, get_peft_model, TaskType

   config = LoraConfig(
       r=8,
       lora_alpha=16,
       target_modules=["q_proj", "v_proj"],
       lora_dropout=0.05,
       task_type=TaskType.CAUSAL_LM,
   )
   model = get_peft_model(base_model, config)
   model.print_trainable_parameters()
   ```
3. Record the trainable parameter count and percentage.

### Part B — Fine-tuning and memory comparison
1. Fine-tune for 100 steps on a small instruction dataset (e.g., a slice of `tatsu-lab/alpaca` or `HuggingFaceH4/ultrachat_200k`).
2. Record peak VRAM with `torch.cuda.max_memory_allocated()` after the first backward pass.
3. If VRAM allows, attempt the same 100-step fine-tune without LoRA (unfreezing all weights) and compare peak VRAM.

## Expected Observations

- **Trainable parameter count drops by 100× or more** — `print_trainable_parameters()` should show 0.1% or less for r=8 on standard LLM attention layers.
- **Peak VRAM with LoRA is significantly lower** than full fine-tuning — often enabling a 7B model on a single 24 GB consumer GPU where full fine-tuning requires 80 GB+.
- Full fine-tuning on the same hardware should OOM; if not, increase the model size until it does — then switch back to LoRA and confirm it fits.
- Training loss should decrease over 100 steps (sanity check that the adapters are learning).

## Why This Matters

LoRA is the entry point to fine-tuning for most practitioners because it makes adapter training accessible without a multi-GPU cluster. Understanding *why* it is memory-efficient — only low-rank matrices carry gradients — is the foundation for evaluating other PEFT methods (QLoRA, DoRA, LoRA+) and for choosing rank r intelligently: higher r captures more adaptation capacity but uses more memory and risks overfitting on small datasets.

## Variant Experiments

- Sweep `r` values (r ∈ {4, 8, 16, 32}) and compare peak VRAM and loss at 200 steps — observe the memory-quality tradeoff.
- Try different `target_modules`: add `"k_proj"`, `"o_proj"`, `"gate_proj"`, `"up_proj"`, `"down_proj"` progressively and observe trainable parameter count and final loss.
- Combine LoRA with 4-bit quantization (QLoRA via `bitsandbytes`): load the base model in 4-bit, apply LoRA, and compare peak VRAM to 16-bit LoRA. Observe whether quality degrades for your task.
