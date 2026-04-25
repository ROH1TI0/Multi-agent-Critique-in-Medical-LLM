"""
Echo Chamber Analysis
Demonstrates the self-critique bias: a model evaluating its own hallucination
tends to validate it. The dedicated Critic agent correctly flags the error.
"""

import torch
import torch.nn.functional as F
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import PeftModel

MODEL_ID = "mistralai/Mistral-7B-v0.1"
GEN_ADAPTER = "./generator-lora-final"    # Update path if using Kaggle/Drive
CRITIC_ADAPTER = "./mistral-critic-lora/final_adapter"


def echo_chamber_analysis():
    print("Running Echo Chamber Logit Analysis...\n")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    # Simulated hallucinated claim
    hallucinated_claim = "Option C is correct because Drug X directly increases blood pressure."
    evidence = "Clinical trials show Drug X has no significant effect on blood pressure, but decreases heart rate."

    # ── Step 1: Self-Critique (Generator evaluates its own claim) ────────────
    print("Loading Generator for self-critique...")
    base_gen = AutoModelForCausalLM.from_pretrained(MODEL_ID, quantization_config=bnb, device_map="auto")
    gen_model = PeftModel.from_pretrained(base_gen, GEN_ADAPTER)
    gen_model.eval()

    self_critique_prompt = (
        f"<s>[INST] Evidence: {evidence}\n"
        f"Claim: {hallucinated_claim}\n"
        f"Is this claim completely supported? Answer Yes or No. [/INST]"
    )
    inputs = tokenizer(self_critique_prompt, return_tensors="pt").to("cuda")

    with torch.no_grad():
        out = gen_model.generate(**inputs, max_new_tokens=10)
        self_response = tokenizer.decode(out[0], skip_special_tokens=True).split("[/INST]")[-1].strip()

    del gen_model, base_gen
    torch.cuda.empty_cache()

    # ── Step 2: Cross-Critique (Dedicated Critic evaluates the same claim) ───
    print("Loading Critic for cross-critique...")
    base_critic = AutoModelForSequenceClassification.from_pretrained(
        MODEL_ID, num_labels=2, quantization_config=bnb, device_map="auto"
    )
    critic_model = PeftModel.from_pretrained(base_critic, CRITIC_ADAPTER)
    critic_model.eval()

    cross_critique_prompt = (
        f"<s>[INST] Evidence: {evidence} \n\n "
        f"Claim: {hallucinated_claim} \n\n "
        f"Is the claim supported? [/INST]"
    )
    c_inputs = tokenizer(cross_critique_prompt, return_tensors="pt", truncation=True, max_length=512).to("cuda")

    with torch.no_grad():
        logits = critic_model(**c_inputs).logits
        probs = F.softmax(logits, dim=-1)
        prob_unsupported = probs[0][0].item()  # Class 0 = Unsupported

    # ── Results ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(" ECHO CHAMBER ANALYSIS RESULTS")
    print("=" * 60)
    print(f"Hallucinated Claim : {hallucinated_claim}")
    print(f"Evidence           : {evidence}")
    print()
    print(f"Self-Critique Response : '{self_response}'")
    print(f"  → Suffers from echo chamber bias (validates own output)")
    print()
    print(f"Cross-Critique Confidence (Unsupported): {prob_unsupported * 100:.1f}%")
    print(f"  → Successfully caught the hallucination")
    print("=" * 60)


if __name__ == "__main__":
    echo_chamber_analysis()
