import random
import numpy as np
from datasets import load_dataset, Dataset
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
    losses,
)
from sentence_transformers.training_args import BatchSamplers

BASE_MODEL = "nreimers/MiniLM-L6-H384-uncased"  # raw model
BATCH_SIZE = 64
TRAIN_SIZE = 20000
N = 2000
SAVE_NAME = "simcse-model"

# Test data (same as baselines.py)
random.seed(42)
ds = load_dataset("nyu-mll/glue", "qqp", split="validation")
pairs = [(r["question1"], r["question2"]) for r in ds if r["label"] == 1]
random.shuffle(pairs)
pairs = pairs[:N]
queries = [p[0] for p in pairs]
docs = [p[1] for p in pairs]
eval_texts = set(queries) | set(docs)


def evaluate(name, model):
    q = model.encode(queries, normalize_embeddings=True, show_progress_bar=False)
    d = model.encode(docs, normalize_embeddings=True, show_progress_bar=False)
    scores = q @ d.T
    correct = np.diag(scores)[:, None]
    ranks = (scores > correct).sum(axis=1) + 1
    print(f"{name:16s} "
          f"Recall@1={np.mean(ranks <= 1):.3f}  "
          f"Recall@5={np.mean(ranks <= 5):.3f}  "
          f"Recall@10={np.mean(ranks <= 10):.3f}  "
          f"MRR={np.mean(1.0 / ranks):.3f}")


# Unlabeled training sentences (no overlap with test)
train_raw = load_dataset("nyu-mll/glue", "qqp", split="train")
sentences = set()
for r in train_raw:
    sentences.add(r["question1"])
    sentences.add(r["question2"])
sentences = [s for s in sentences if s not in eval_texts]
random.Random(0).shuffle(sentences)
sentences = sentences[:TRAIN_SIZE]
print(f"Training sentences: {len(sentences)}")

# SimCSE: anchor and positive are the same sentence; dropout makes them differ
train_ds = Dataset.from_dict({"anchor": sentences, "positive": sentences})

model = SentenceTransformer(BASE_MODEL)
evaluate("Raw, untrained", model)

loss = losses.MultipleNegativesRankingLoss(model)
args = SentenceTransformerTrainingArguments(
    output_dir="output",
    num_train_epochs=1,
    per_device_train_batch_size=BATCH_SIZE,
    learning_rate=5e-5,
    warmup_steps=0.1,
    batch_sampler=BatchSamplers.NO_DUPLICATES,
    logging_steps=50,
    save_strategy="no",
    report_to="none",
)
trainer = SentenceTransformerTrainer(
    model=model, args=args, train_dataset=train_ds, loss=loss
)
trainer.train()

evaluate("After SimCSE", model)
model.save(SAVE_NAME)
print(f"Model saved in folder: {SAVE_NAME}")