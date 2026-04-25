"""
Generator Agent Training
Fine-tunes Mistral-7B on MedMCQA for clinical multiple-choice reasoning.
Uses QLoRA to keep VRAM under 12GB.
"""

import os
import gc
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM, SFTConfig

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

MODEL_ID = "mistralai/Mistral-7B-v0.1"
OUTPUT_DIR = "./generator-lora"
ADAPTER_DIR = "./generator-lora-final"


def clear_vram():
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, "data") and torch.is_tensor(obj.data)):
                del obj
        except Exception:
            pass
    torch.cuda.empty_cache()
    gc.collect()


def format_example(ex, tokenizer):
    answer_map = {0: "A", 1: "B", 2: "C", 3: "D"}
    q, a, b, c, d = ex["question"], ex["opa"], ex["opb"], ex["opc"], ex["opd"]
    correct = answer_map.get(ex["cop"], "A")
    exp = ex.get("exp") or f"Option {correct} is medically established."
    text = (
        "<s>[INST] You are a medical expert. Read the question, select the correct answer, "
        "and explain your reasoning.\n"
        f"Question: {q}\nA) {a}\nB) {b}\nC) {c}\nD) {d} [/INST] "
        f"Answer: {correct}\nExplanation: {exp}{tokenizer.eos_token}"
    )
    return {"text": text}


def train_generator():
    clear_vram()

    print("Loading MedMCQA...")
    dataset = load_dataset("medmcqa", split="train[:5000]")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    train_dataset = dataset.map(
        lambda ex: format_example(ex, tokenizer),
        remove_columns=dataset.column_names,
    )

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map={"": 0},
        attn_implementation="eager",
        torch_dtype=torch.float16,
    )
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    collator = DataCollatorForCompletionOnlyLM("[/INST]", tokenizer=tokenizer)

    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        learning_rate=2e-4,
        num_train_epochs=1,
        optim="paged_adamw_8bit",
        fp16=True,
        bf16=False,
        save_strategy="epoch",
        report_to="none",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_seq_length=512,
        dataset_text_field="text",
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        args=training_args,
        data_collator=collator,
    )

    trainer.train()
    trainer.model.save_pretrained(ADAPTER_DIR)
    tokenizer.save_pretrained(ADAPTER_DIR)
    print("✅ Generator Training Complete. Adapter saved to:", ADAPTER_DIR)


if __name__ == "__main__":
    train_generator()
