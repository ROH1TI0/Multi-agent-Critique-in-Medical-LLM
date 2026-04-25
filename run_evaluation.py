"""
Full Evaluation Pipeline
Runs Generator + Self-Critique + Cross-Critique on 100 MedMCQA validation questions.
Performs threshold sweep (50%–90%) and generates comparison visualizations.
"""

import os
import gc
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import PeftModel

MODEL_ID = "mistralai/Mistral-7B-v0.1"
GEN_ADAPTER = "./generator-lora-final"
CRITIC_ADAPTER = "./mistral-critic-lora/final_adapter"
NUM_EVAL_QUESTIONS = 100
OPTIMAL_THRESHOLD = 0.80

bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)


def clear_vram():
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, "data") and torch.is_tensor(obj.data)):
                del obj
        except Exception:
            pass
    torch.cuda.empty_cache()
    gc.collect()


def run_final_evaluation():
    print(f"Loading MedMCQA Validation Set ({NUM_EVAL_QUESTIONS} questions)...")
    dataset = load_dataset("medmcqa", split=f"validation[:{NUM_EVAL_QUESTIONS}]")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    answer_map = {0: "A", 1: "B", 2: "C", 3: "D"}

    results = []

    # ── Stage 1: Generator & Self-Critique ───────────────────────────────────
    print("\n--- STAGE 1: Generator (Answer) + Self-Critique ---")
    base_gen = AutoModelForCausalLM.from_pretrained(MODEL_ID, quantization_config=bnb, device_map="auto")
    gen_model = PeftModel.from_pretrained(base_gen, GEN_ADAPTER)
    gen_model.eval()

    for ex in dataset:
        q, a, b, c, d = ex["question"], ex["opa"], ex["opb"], ex["opc"], ex["opd"]
        correct_ans = answer_map.get(ex["cop"], "A")

        # Generate answer
        gen_prompt = (
            f"<s>[INST] You are a medical expert. Answer this:\n"
            f"{q}\nA) {a}\nB) {b}\nC) {c}\nD) {d} [/INST]"
        )
        inputs = tokenizer(gen_prompt, return_tensors="pt").to("cuda")
        out = gen_model.generate(**inputs, max_new_tokens=50, pad_token_id=tokenizer.eos_token_id)
        generated_text = tokenizer.decode(out[0], skip_special_tokens=True).split("[/INST]")[-1].strip()
        is_correct = correct_ans in generated_text[:15]

        # Self-critique
        self_crit_prompt = (
            f"<s>[INST] You answered: {generated_text}. "
            f"Is this answer completely supported and factually correct? Answer Yes or No. [/INST]"
        )
        sc_inputs = tokenizer(self_crit_prompt, return_tensors="pt").to("cuda")
        sc_out = gen_model.generate(**sc_inputs, max_new_tokens=10, pad_token_id=tokenizer.eos_token_id)
        sc_text = tokenizer.decode(sc_out[0], skip_special_tokens=True).split("[/INST]")[-1].strip().lower()
        self_flagged = "no" in sc_text

        results.append({
            "question": q,
            "correct_ans": correct_ans,
            "gen_text": generated_text,
            "is_correct": is_correct,
            "self_flagged": self_flagged,
        })

    del gen_model, base_gen
    clear_vram()

    # ── Stage 2: Cross-Critique (Save Probabilities) ─────────────────────────
    print("\n--- STAGE 2: Cross-Critique (Threshold Sweep) ---")
    base_critic = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID, num_labels=2, quantization_config=bnb, device_map="auto"
    )
    critic_model = PeftModel.from_pretrained(base_critic, CRITIC_ADAPTER)
    critic_model.eval()

    for res in results:
        cc_prompt = (
            f"<s>[INST] Question context: {res['question']}\n\n"
            f"Evaluate this generated answer: {res['gen_text']}\n\n"
            f"Is this answer completely supported? [/INST]"
        )
        cc_inputs = tokenizer(cc_prompt, return_tensors="pt", truncation=True, max_length=384).to("cuda")

        with torch.no_grad():
            logits = critic_model(**cc_inputs).logits
            probs = F.softmax(logits, dim=-1)
            res["prob_unsupported"] = probs[0][0].item()  # Class 0 = Unsupported

    del critic_model, base_critic
    clear_vram()

    # ── Stage 3: Metrics ─────────────────────────────────────────────────────
    total_samples = len(results)
    total_correct = sum(1 for r in results if r["is_correct"])
    total_incorrect = total_samples - total_correct

    sc_true_flags  = sum(1 for r in results if not r["is_correct"] and r["self_flagged"])
    sc_false_flags = sum(1 for r in results if r["is_correct"] and r["self_flagged"])
    sc_true_negatives = total_correct - sc_false_flags

    sc_cr  = (sc_true_flags / total_incorrect * 100) if total_incorrect > 0 else 0
    sc_fcr = (sc_false_flags / total_correct * 100) if total_correct > 0 else 0
    sc_accuracy = ((sc_true_flags + sc_true_negatives) / total_samples) * 100

    print("\n" + "=" * 70)
    print(" FINAL EVALUATION METRICS")
    print("=" * 70)
    print(f"Base Generator Accuracy : {total_correct}%  (Hallucinations: {total_incorrect})\n")
    print("--- SELF-CRITIQUE (Baseline) ---")
    print(f"Overall Accuracy : {sc_accuracy:.1f}%")
    print(f"CR: {sc_cr:.1f}%  |  FCR: {sc_fcr:.1f}%\n")

    print("--- CROSS-CRITIQUE (Threshold Sweep) ---")
    print(f"{'Threshold':<12} {'Accuracy':<16} {'CR':<22} {'FCR'}")
    print("-" * 70)

    thresholds = [0.50, 0.60, 0.70, 0.80, 0.85, 0.90]
    sweep_cr, sweep_fcr, sweep_acc = [], [], []

    for t in thresholds:
        t_true  = sum(1 for r in results if not r["is_correct"] and r["prob_unsupported"] > t)
        t_false = sum(1 for r in results if r["is_correct"] and r["prob_unsupported"] > t)
        t_tn    = total_correct - t_false

        t_cr  = (t_true / total_incorrect * 100) if total_incorrect > 0 else 0
        t_fcr = (t_false / total_correct * 100) if total_correct > 0 else 0
        t_acc = ((t_true + t_tn) / total_samples) * 100

        sweep_cr.append(t_cr)
        sweep_fcr.append(t_fcr)
        sweep_acc.append(t_acc)

        marker = " ← optimal" if t == OPTIMAL_THRESHOLD else ""
        print(f"  >{int(t*100)}%{'':<8} {t_acc:.1f}%{'':<10} {t_cr:.1f}%{'':<16} {t_fcr:.1f}%{marker}")

    print("=" * 70)

    # ── Stage 4: Visualizations ───────────────────────────────────────────────
    idx_optimal = thresholds.index(OPTIMAL_THRESHOLD)
    cc_opt_cr  = sweep_cr[idx_optimal]
    cc_opt_fcr = sweep_fcr[idx_optimal]
    cc_opt_acc = sweep_acc[idx_optimal]

    os.makedirs("figures", exist_ok=True)

    # Plot 1 — Overall Accuracy Comparison
    plt.figure(figsize=(8, 5))
    labels = ["Base Generator", "Self-Critique", f"Cross-Critique ({int(OPTIMAL_THRESHOLD*100)}%)"]
    values = [total_correct, sc_accuracy, cc_opt_acc]
    colors = ["#cccccc", "#ff9999", "#66b3ff"]
    bars = plt.bar(labels, values, color=colors, edgecolor="black")
    plt.ylim(0, 100)
    plt.ylabel("Overall Accuracy (%)", fontweight="bold")
    plt.title("Overall Verification Accuracy Comparison", fontweight="bold", pad=15)
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval + 2, f"{yval:.1f}%", ha="center", va="bottom", fontweight="bold")
    plt.tight_layout()
    plt.savefig("figures/accuracy_comparison.png", dpi=150)
    plt.show()

    # Plot 2 — CR vs FCR Showdown
    plt.figure(figsize=(9, 5))
    bar_width = 0.35
    index = np.arange(2)
    cr_values  = [sc_cr, cc_opt_cr]
    fcr_values = [sc_fcr, cc_opt_fcr]
    bar1 = plt.bar(index, cr_values, bar_width, label="Correction Rate (Caught Errors)", color="#4CAF50", edgecolor="black")
    bar2 = plt.bar(index + bar_width, fcr_values, bar_width, label="False Correction Rate (Collateral)", color="#F44336", edgecolor="black")
    plt.xlabel("Verification Method", fontweight="bold")
    plt.ylabel("Percentage (%)", fontweight="bold")
    plt.title(f"Self-Critique vs Cross-Critique (>{int(OPTIMAL_THRESHOLD*100)}% Threshold)", fontweight="bold", pad=15)
    plt.xticks(index + bar_width / 2, ["Self-Critique", "Cross-Critique"])
    plt.legend()
    for bar in bar1 + bar2:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval + 0.5, f"{yval:.1f}%", ha="center", va="bottom", fontweight="bold")
    plt.tight_layout()
    plt.savefig("figures/cr_vs_fcr.png", dpi=150)
    plt.show()

    # Plot 3 — Threshold Sweep Line Graph
    plt.figure(figsize=(9, 5))
    plt.plot(thresholds, sweep_cr,  marker="o", linestyle="-",  color="#4CAF50", linewidth=2, label="Correction Rate (CR)")
    plt.plot(thresholds, sweep_fcr, marker="s", linestyle="--", color="#F44336", linewidth=2, label="False Correction Rate (FCR)")
    plt.axvline(x=OPTIMAL_THRESHOLD, color="grey", linestyle=":", linewidth=2, label=f"Optimal Threshold ({int(OPTIMAL_THRESHOLD*100)}%)")
    plt.title("Cross-Critique Threshold Calibration", fontweight="bold", pad=15)
    plt.xlabel("Confidence Threshold", fontweight="bold")
    plt.ylabel("Percentage (%)", fontweight="bold")
    plt.xticks(thresholds, [f"{int(t * 100)}%" for t in thresholds])
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig("figures/threshold_sweep.png", dpi=150)
    plt.show()

    print("\n✅ Figures saved to ./figures/")


if __name__ == "__main__":
    run_final_evaluation()
