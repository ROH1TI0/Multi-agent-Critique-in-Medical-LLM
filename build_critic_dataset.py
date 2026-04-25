"""
Critic Dataset Builder
Constructs a triplet dataset from PubMedQA for binary claim verification.

Label 1 — Ground truth (claim + matching abstract)
Label 0 (Type A) — TF-IDF hard negative (related but mismatched abstract)
Label 0 (Type B) — Subtle hallucination via number/negation/directional mutation
"""

import json
import random
import re
import numpy as np
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

TRAIN_OUTPUT = "pubmedqa_critic_train_v2.jsonl"
TEST_OUTPUT = "pubmedqa_critic_test_v2.jsonl"
TARGET_CLAIMS = 25000
TRAIN_SPLIT = 0.9
BATCH_SIZE = 1000


def corrupt_claim(claim):
    """Mutates a true medical claim into a subtle hallucination."""
    corrupted = claim
    mutation_type = random.choice(["number", "negation", "directional"])

    if mutation_type == "number":
        def change_num(match):
            num = float(match.group(0))
            return str(round(num * random.choice([0.1, 2.0, 5.0, 10.0]), 2))
        corrupted = re.sub(r'\b\d+(\.\d+)?\b', change_num, claim)

    elif mutation_type == "negation":
        if " not " in claim:
            corrupted = claim.replace(" not ", " ")
        else:
            words = claim.split()
            if len(words) > 3:
                insert_idx = random.randint(1, len(words) - 1)
                words.insert(insert_idx, "not")
                corrupted = " ".join(words)

    elif mutation_type == "directional":
        swaps = {
            "increase": "decrease", "decrease": "increase",
            "higher": "lower", "lower": "higher",
            "improves": "worsens", "worsens": "improves",
            "positive": "negative", "negative": "positive",
        }
        for word, replacement in swaps.items():
            if word in corrupted:
                corrupted = corrupted.replace(word, replacement)
                break

    if corrupted == claim:
        corrupted = "No evidence suggests: " + claim.lower()

    return corrupted


def build_critic_dataset():
    print("----- Loading PubMedQA Dataset -----")
    dataset = load_dataset("pubmed_qa", "pqa_unlabeled", split="train")

    contexts = [" ".join(ex["context"]["contexts"]) for ex in dataset]
    claims = [ex["question"] for ex in dataset]

    unique_claims = list(set(claims))
    random.shuffle(unique_claims)

    target_claims = min(TARGET_CLAIMS, len(unique_claims))
    active_claims = unique_claims[:target_claims]

    split_idx = int(len(active_claims) * TRAIN_SPLIT)
    train_claims_set = set(active_claims[:split_idx])

    claim_to_context_idx = {claim: i for i, claim in enumerate(claims)}

    print("----- Vectorizing for TF-IDF Hard Negatives -----")
    vectorizer = TfidfVectorizer(stop_words="english", max_features=50000)
    tfidf_contexts = vectorizer.fit_transform(contexts)
    tfidf_claims = vectorizer.transform(active_claims)

    train_data, test_data = [], []

    print("----- Generating Triplet Examples -----")
    for start_idx in tqdm(range(0, len(active_claims), BATCH_SIZE)):
        end_idx = min(start_idx + BATCH_SIZE, len(active_claims))
        batch_similarities = (tfidf_claims[start_idx:end_idx] * tfidf_contexts.T).toarray()

        for i, row_scores in enumerate(batch_similarities):
            claim = active_claims[start_idx + i]
            true_idx = claim_to_context_idx[claim]
            true_context = contexts[true_idx]

            target_list = train_data if claim in train_claims_set else test_data

            # Label 1: Ground Truth
            target_list.append({
                "text": f"<s>[INST] Evidence: {true_context} \n\n Claim: {claim} \n\n Is the claim supported? [/INST]",
                "label": 1,
            })

            # Label 0 (Type A): TF-IDF Hard Negative
            top_50_indices = np.argpartition(row_scores, -50)[-50:]
            top_50_indices = top_50_indices[np.argsort(row_scores[top_50_indices])][::-1]
            mismatched_idx = random.choice(top_50_indices[10:])

            target_list.append({
                "text": f"<s>[INST] Evidence: {contexts[mismatched_idx]} \n\n Claim: {claim} \n\n Is the claim supported? [/INST]",
                "label": 0,
            })

            # Label 0 (Type B): Subtle Hallucination
            hallucinated_claim = corrupt_claim(claim)
            target_list.append({
                "text": f"<s>[INST] Evidence: {true_context} \n\n Claim: {hallucinated_claim} \n\n Is the claim supported? [/INST]",
                "label": 0,
            })

    print("\n----- Saving Datasets -----")
    random.shuffle(train_data)
    random.shuffle(test_data)

    with open(TRAIN_OUTPUT, "w") as f:
        for item in train_data:
            f.write(json.dumps(item) + "\n")

    with open(TEST_OUTPUT, "w") as f:
        for item in test_data:
            f.write(json.dumps(item) + "\n")

    print(f"✅ Done. {len(train_data)} training pairs, {len(test_data)} test pairs.")
    print(f"   Train → {TRAIN_OUTPUT}")
    print(f"   Test  → {TEST_OUTPUT}")


if __name__ == "__main__":
    build_critic_dataset()
