# eagle-t4-colab-benchmarks

> A plug-and-play companion to [SafeAILab/EAGLE](https://github.com/SafeAILab/EAGLE) — run speculative decoding benchmarks instantly on Google Colab's free T4 GPU.

The official EAGLE repo is a research codebase. Getting it running on Colab requires resolving dependency conflicts, patching incompatible modules, and navigating tokenizer quirks across model families. This repo does all of that for you.

Open a notebook, run it, get results.

---

## What This Repo Does

- Wraps the official EAGLE repo with all patches and fixes pre-applied
- Handles dependency pinning and monkey-patching automatically within each notebook
- Covers 7 popular open-source LLMs across EAGLE and EAGLE-3 variants
- Runs entirely within Colab's **free T4 GPU tier** using 8-bit quantization
- Reports speedup, TPS (tokens/sec), and TVD-based draft alignment metrics

---

## Why the T4 GPU

The T4 was chosen deliberately — not just because it's free.

With 15GB VRAM, the T4 genuinely mimics a **resource-constrained environment**. It sits in the same class of hardware you'd encounter on edge devices and on-premise inference servers where you can't assume abundant GPU memory or high-end compute. Running EAGLE benchmarks on the T4 gives you a realistic picture of speculative decoding performance under real constraints, not just on datacenter-class hardware.

If your goal is to deploy EAGLE on an edge device or alongside other workloads, these benchmarks give you a **concrete headstart**. You'll already know which models fit, what speedups are achievable under 8-bit quantization, and where the memory ceiling is — before you spend time tuning for production. The jump from T4 to your target hardware is a calibration exercise, not a discovery exercise.

---

## Notebooks

| # | Notebook | Base Model | EAGLE Model | Type | Gated |
|---|----------|------------|-------------|------|-------|
| 1 | `bench_llama31_8b_e1_1.ipynb` | meta-llama/Llama-3.1-8B-Instruct | yuhuili/EAGLE-LLaMA3.1-Instruct-8B | EAGLE | ✅ Yes |
| 2 | `bench_llama31_8b_e3_2.ipynb` | meta-llama/Llama-3.1-8B-Instruct | yuhuili/EAGLE3-LLaMA3.1-Instruct-8B | EAGLE-3 | ✅ Yes |
| 3 | `bench_llama3_8b_e1_3.ipynb` | meta-llama/Meta-Llama-3-8B-Instruct | yuhuili/EAGLE-LLaMA3-Instruct-8B | EAGLE | ✅ Yes |
| 4 | `bench_llama2_7b_e1_4.ipynb` | meta-llama/Llama-2-7b-chat-hf | yuhuili/EAGLE-llama2-chat-7B | EAGLE | ✅ Yes |
| 5 | `bench_qwen2_7b_e1_5.ipynb` | Qwen/Qwen2-7B-Instruct | yuhuili/EAGLE-Qwen2-7B-Instruct | EAGLE | No |
| 6 | `bench_qwen3_8b_e3_6.ipynb` | Qwen/Qwen3-8B | AngelSlim/Qwen3-8B_eagle3 | EAGLE-3 | No |
| 7 | `bench_deepseek_8b_e3_7.ipynb` | deepseek-ai/DeepSeek-R1-Distill-Llama-8B | yuhuili/EAGLE3-DeepSeek-R1-Distill-LLaMA-8B | EAGLE-3 | No |

---

## Quickstart

### For non-gated models (Qwen, DeepSeek)
1. Open any notebook in Colab
2. Set runtime to **T4 GPU** (Runtime → Change runtime type)
3. Run the cell

### For gated models (LLaMA 2, LLaMA 3, LLaMA 3.1)
1. Request access on Hugging Face:
   - [meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)
   - [meta-llama/Meta-Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct)
   - [meta-llama/Llama-2-7b-chat-hf](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf)
2. In Colab, open **Secrets** (🔑 left sidebar) → add `HF_TOKEN` with your Hugging Face token
3. Run the cell

That's it. Each notebook self-installs all dependencies and applies all patches automatically.

---

## What Each Notebook Reports

**Performance**
- Baseline TPS (tokens/sec) — standard autoregressive generation
- EAGLE TPS (tokens/sec) — speculative decoding
- Speedup ratio and time saved per 100 tokens

**Divergence Analysis**
- Average accepted tokens per step (τ)
- Token acceptance rate (α)
- Total Variation Distance (TVD) — derived from observed speedup via speculative decoding theory

---

## What's Being Patched and Why

The official EAGLE repo targets a research environment. Running it on Colab's current image requires three fixes that each notebook applies automatically:

| Fix | Why |
|-----|-----|
| `transformers==4.53.1` `list_repo_templates` monkey-patch | Colab's default transformers throws a 404 error on older model repos that lack an `additional_chat_templates` folder |
| `cnets.py` `draft_vocab_size` fallback | Some EAGLE weight configs omit this attribute; falling back to `vocab_size` is safe |
| `modeling_qwen3_kv.py` stub | The Qwen3 KV module imports transformers internals that don't exist in 4.53.1; stubbing it out prevents import failures on non-Qwen3 models |

---

## Hardware

| | |
|--|--|
| GPU | NVIDIA T4 |
| VRAM | 15 GB |
| Quantization | 8-bit (BitsAndBytes) |
| Platform | Google Colab free tier |

> **Note:** Vicuna 7B is intentionally excluded. It loads successfully but consistently OOMs during EAGLE generation on the T4 — the combined memory pressure of base model + EAGLE head + KV cache during speculative decoding exceeds 15GB.

---
## Results

### LLaMA 3.1 8B — EAGLE
![Deepseek 8B EAGLE 3](bench_deepseek_8b_e3_7.ipynb)

---
## References

- [EAGLE paper](https://arxiv.org/abs/2401.15077)
- [EAGLE-2 paper](https://arxiv.org/abs/2406.16858)
- [SafeAILab/EAGLE (official repo)](https://github.com/SafeAILab/EAGLE)
