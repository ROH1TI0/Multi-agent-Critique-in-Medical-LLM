"""
Critic Agent Training
Fine-tunes Mistral-7B as a binary sequence classifier on PubMedQA triplets.
Run build_critic_dataset.py first to generate the training data.
"""

import os
import gc
import torch
import numpy as np
import evaluate
from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

MODEL_NAME = "mistralai/Mistral-7B-v0.1"
OUTPUT_DIR = "./mistral-critic-lora"
FINAL_ADAPTER_DIR = "./mistral-critic-lora/final_adapter"
TRAIN_FILE = "pubmedqa_critic_train_v2.jsonl"
TEST_FILE = "pubmedqa_critic_test_v2.jsonl"


def clear_vram():
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, "data") and torch.is_tensor(obj.data)):
                del obj
        except Exception:
            pass
    torch.cuda.empty_cache()
    gc.collect()


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    accuracy = evaluate.load("accuracy")
    return accuracy.compute(
        predictions=np.argmax(predictions, axis=1),
        references=labels,
    )


def train_critic():
    clear_vram()

    dataset = load_dataset(
        "json",
        data_files={"train": TRAIN_FILE, "test": TEST_FILE},
    )
    # Subset for faster iteration — remove to use full dataset
    dataset["train"] = dataset["train"].select(range(5000))

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
        device_map={"": 0},
        quantization_config=bnb_config,
        attn_implementation="eager",
        torch_dtype=torch.float16,
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model = prepare_model_for_kbit_training(model)

    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="SEQ_CLS",
    )
    model = get_peft_model(model, peft_config)

    def tokenize_function(examples):
        tokenized = tokenizer(examples["text"], truncation=True, max_length=384)
        tokenized["label"] = [int(label) for label in examples["label"]]
        return tokenized

    tokenized_train = (
        dataset["train"]
        .map(tokenize_function, batched=True)
        .select_columns(["input_ids", "attention_mask", "label"])
    )
    tokenized_test = (
        dataset["test"]
        .map(tokenize_function, batched=True)
        .select_columns(["input_ids", "attention_mask", "label"])
    )

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        num_train_epochs=1,
        optim="paged_adamw_8bit",
        fp16=True,
        bf16=False,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_test,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.model.save_pretrained(FINAL_ADAPTER_DIR)
    tokenizer.save_pretrained(FINAL_ADAPTER_DIR)
    print("✅ Critic Training Complete. Adapter saved to:", FINAL_ADAPTER_DIR)


if __name__ == "__main__":
    train_critic()
