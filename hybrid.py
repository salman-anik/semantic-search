import random
import numpy as np
from datasets import load_dataset
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

N = 2000
MODEL_PATH = "simcse-model"   # the model you trained with train.py

# Same data order as baselines.py and train.py (seed 42)
random.seed(42)
ds = load_dataset("nyu-mll/glue", "qqp", split="validation")
pairs = [(r["question1"], r["question2"]) for r in ds if r["label"] == 1]
random.shuffle(pairs)
test_pairs = pairs[:N]         # the same 2,000 test pairs as before
tune_pairs = pairs[N:2 * N]    # a different 2,000 pairs, used only to pick the mix

model = SentenceTransformer(MODEL_PATH)


def get_scores(pair_list):
    queries = [p[0] for p in pair_list]
    docs = [p[1] for p in pair_list]
    # BM25 scores
    bm25 = BM25Okapi([d.lower().split() for d in docs])
    bm25_scores = np.array([bm25.get_scores(q.lower().split()) for q in queries])
    # Our model's scores
    q = model.encode(queries, normalize_embeddings=True, show_progress_bar=False)
    d = model.encode(docs, normalize_embeddings=True, show_progress_bar=False)
    return bm25_scores, q @ d.T


def zscore(m):
    # put each query's scores on the same scale
    return (m - m.mean(axis=1, keepdims=True)) / (m.std(axis=1, keepdims=True) + 1e-9)


def metrics(scores):
    correct = np.diag(scores)[:, None]
    ranks = (scores > correct).sum(axis=1) + 1
    return (np.mean(ranks <= 1), np.mean(ranks <= 5),
            np.mean(ranks <= 10), np.mean(1.0 / ranks))


def show(name, scores):
    r1, r5, r10, mrr = metrics(scores)
    print(f"{name:26s} Recall@1={r1:.3f}  Recall@5={r5:.3f}  "
          f"Recall@10={r10:.3f}  MRR={mrr:.3f}")


# Step A: choose the mix weight on the TUNING pairs
print("Scoring tuning pairs ...")
bm_t, md_t = get_scores(tune_pairs)
bm_t, md_t = zscore(bm_t), zscore(md_t)

best_alpha, best_mrr = 0.0, -1.0
print("\nTuning (alpha = weight of our model, 1 - alpha = weight of BM25):")
for alpha in np.arange(0.0, 1.01, 0.1):
    mrr = metrics(alpha * md_t + (1 - alpha) * bm_t)[3]
    print(f"  alpha={alpha:.1f}  MRR={mrr:.3f}")
    if mrr > best_mrr:
        best_alpha, best_mrr = alpha, mrr
print(f"\nChosen alpha = {best_alpha:.1f}")

# Step B: final result on the TEST pairs
print("\nScoring test pairs ...")
bm_s, md_s = get_scores(test_pairs)
bm_s, md_s = zscore(bm_s), zscore(md_s)

print("\n=== Test results (same 2,000 pairs as the earlier table) ===")
show("BM25 only", bm_s)
show("Our model only", md_s)
show(f"Hybrid (alpha={best_alpha:.1f})", best_alpha * md_s + (1 - best_alpha) * bm_s)