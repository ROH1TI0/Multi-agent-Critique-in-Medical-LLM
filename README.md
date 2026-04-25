# Multi-Agent Cross-Critique LLM Architecture

A dual-agent system for medical question answering that separates **answer generation** from **factual verification** to mitigate hallucinations in clinical reasoning tasks.

---

## Overview

Large language models suffer from an "echo chamber" bias when asked to critique their own outputs — they tend to validate what they generated. This project addresses that by training two **independent, specialized agents**:

- **Generator Agent** — Fine-tuned on [MedMCQA](https://huggingface.co/datasets/medmcqa) for clinical multiple-choice reasoning
- **Critic Agent** — Fine-tuned on [PubMedQA](https://huggingface.co/datasets/pubmed_qa) for binary claim verification (Supported / Unsupported)

The Critic evaluates the Generator's outputs independently, using a **threshold-calibrated confidence score** to flag hallucinations without over-correcting on valid answers.

---

## Architecture

```
┌─────────────────────┐        ┌──────────────────────────┐
│   Generator Agent   │        │      Critic Agent        │
│  Mistral-7B + LoRA  │──────▶│  Mistral-7B + LoRA (SEQ) │
│  (Causal LM)        │        │  (Sequence Classification)│
│  Trained: MedMCQA   │        │  Trained: PubMedQA        │
└─────────────────────┘        └──────────────────────────┘
         │                                  │
         ▼                                  ▼
   Generated Answer              Confidence Score (0–1)
                                 Threshold @ 0.80 → Flag
```

---

## Key Results

| Metric | Self-Critique (Baseline) | Cross-Critique (80% threshold) |
|---|---|---|
| Correction Rate (CR) | Low | **4x improvement** |
| False Correction Rate (FCR) | High | Significantly reduced |
| VRAM Usage | — | **< 12GB** (QLoRA) |

---

## Project Structure

```
multi-agent-cross-critique/
├── generator/
│   └── train_generator.py       # Fine-tune Generator on MedMCQA
├── critic/
│   ├── build_critic_dataset.py  # Construct PubMedQA triplet dataset
│   └── train_critic.py          # Fine-tune Critic for binary classification
├── evaluation/
│   ├── echo_chamber_analysis.py # Demonstrate self-critique bias
│   └── run_evaluation.py        # Full threshold sweep + visualizations
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Quickstart

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the Generator

```bash
python generator/train_generator.py
```

Trains Mistral-7B with QLoRA on 5,000 MedMCQA samples. Saves adapter to `./generator-lora-final`.

### 3. Build the Critic Dataset

```bash
python critic/build_critic_dataset.py
```

Generates triplet examples (ground truth, TF-IDF hard negatives, subtle hallucinations) from PubMedQA. Outputs:
- `pubmedqa_critic_train_v2.jsonl`
- `pubmedqa_critic_test_v2.jsonl`

### 4. Train the Critic

```bash
python critic/train_critic.py
```

Fine-tunes Mistral-7B as a sequence classifier using QLoRA. Saves adapter to `./mistral-critic-lora/final_adapter`.

### 5. Run Evaluation

```bash
# Echo chamber demonstration
python evaluation/echo_chamber_analysis.py

# Full threshold sweep on 100 MedMCQA validation questions
python evaluation/run_evaluation.py
```

---

## Technical Details

### Fine-Tuning (Both Agents)
- **Base Model:** `mistralai/Mistral-7B-v0.1`
- **Quantization:** QLoRA — 4-bit NF4, double quantization
- **LoRA Config:** `r=8`, `alpha=16`, targeting all projection layers
- **Optimizer:** `paged_adamw_8bit`
- **VRAM:** < 12GB (tested on Kaggle T4)

### Critic Dataset Construction
Three label types per PubMedQA claim:
1. **Label 1** — Ground truth (claim + matching abstract)
2. **Label 0 (Type A)** — TF-IDF hard negative (related but mismatched abstract)
3. **Label 0 (Type B)** — Subtle hallucination via number/negation/directional mutation

### Threshold Calibration
The Critic outputs a softmax probability. A sweep over thresholds (50%–90%) identifies **80%** as optimal — maximizing Correction Rate while minimizing False Correction Rate.

---

## Requirements

See `requirements.txt`. Designed to run on a single GPU (Kaggle T4 / Google Colab).

---

## Datasets

| Dataset | Usage | Link |
|---|---|---|
| MedMCQA | Generator training + evaluation | [HuggingFace](https://huggingface.co/datasets/medmcqa) |
| PubMedQA (pqa_unlabeled) | Critic dataset construction | [HuggingFace](https://huggingface.co/datasets/pubmed_qa) |

---

## Acknowledgements

Built with [HuggingFace Transformers](https://github.com/huggingface/transformers), [PEFT](https://github.com/huggingface/peft), [TRL](https://github.com/huggingface/trl), and [BitsAndBytes](https://github.com/TimDettmers/bitsandbytes).
