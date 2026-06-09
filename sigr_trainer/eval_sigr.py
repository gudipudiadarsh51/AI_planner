"""
SIGR Model Evaluation
Computes Hits@n and MRR (Section VI-C of the paper)
on held-out test data from BigQuery curated tables.

Usage:
    python eval_sigr.py

Requires:
    - Trained model artifacts in gs://yelp-sigr-training/models/sigr_v1/
    - GCS credentials set via GOOGLE_APPLICATION_CREDENTIALS
"""

import os
import io
import json
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pyarrow.fs as pafs
from google.cloud import storage
from collections import defaultdict

# ── Config ────────────────────────────────────────────────────────────────
BUCKET       = "yelp-sigr-training"
DATA_PREFIX  = "sigr-training"
MODEL_PREFIX = "models/sigr_v1"
MAX_GROUP_SIZE = 10

# Evaluation metrics from paper (Section VI-C)
TOP_N_VALUES = [1, 5, 10, 15, 20]

# ── Load model artifacts from GCS ─────────────────────────────────────────
print("Loading model artifacts from GCS...")
client = storage.Client()
gcs_bucket = client.bucket(BUCKET)

def load_numpy_from_gcs(blob_name):
    blob = gcs_bucket.blob(blob_name)
    buf = io.BytesIO()
    blob.download_to_file(buf)
    buf.seek(0)
    return np.load(buf)

def load_json_from_gcs(blob_name):
    blob = gcs_bucket.blob(blob_name)
    return json.loads(blob.download_as_text())

user_emb          = load_numpy_from_gcs(f"{MODEL_PREFIX}/user_embeddings.npy")
item_emb          = load_numpy_from_gcs(f"{MODEL_PREFIX}/item_embeddings.npy")
social_influence  = load_numpy_from_gcs(f"{MODEL_PREFIX}/social_influence.npy")
user2idx          = load_json_from_gcs(f"{MODEL_PREFIX}/user2idx.json")
item2idx          = load_json_from_gcs(f"{MODEL_PREFIX}/item2idx.json")
group2idx         = load_json_from_gcs(f"{MODEL_PREFIX}/group2idx.json")

# Reverse mappings
idx2item = {int(v): k for k, v in item2idx.items()}
idx2user = {int(v): k for k, v in user2idx.items()}

N_USERS = len(user2idx)
N_ITEMS = len(item2idx)

print(f"  User embeddings:    {user_emb.shape}")
print(f"  Item embeddings:    {item_emb.shape}")
print(f"  Social influence:   {social_influence.shape}")
print(f"  Users: {N_USERS} | Items: {N_ITEMS}")

# ── Load interaction data ──────────────────────────────────────────────────
print("\nLoading interaction data...")
fs = pafs.GcsFileSystem()

def load_parquet(blob_prefix, columns=None):
    dataset = pq.ParquetDataset(
        f"{BUCKET}/{DATA_PREFIX}/{blob_prefix}",
        filesystem=fs
    )
    return dataset.read(columns=columns).to_pandas()

interactions = load_parquet("user_business_interactions", columns=[
    "user_id", "business_id", "review_id", "bgem_edge_weight"
])
group_edges = load_parquet("group_item_edges", columns=[
    "group_id", "item_id", "group_size"
])

print(f"  Interactions: {len(interactions)}")
print(f"  Group edges:  {len(group_edges)}")

# ── Build group member map (same as training) ─────────────────────────────
print("\nBuilding group member map...")
interactions["user_idx"] = interactions["user_id"].map(
    lambda x: user2idx.get(str(x))
)
interactions["item_idx"] = interactions["business_id"].map(
    lambda x: item2idx.get(str(x))
)

item_to_users = interactions.dropna(subset=["item_idx"]).groupby(
    "item_idx"
)["user_idx"].apply(list).to_dict()

group_edges["group_idx"] = group_edges["group_id"].map(
    lambda x: group2idx.get(str(x))
)
group_edges["item_idx"] = group_edges["item_id"].map(
    lambda x: item2idx.get(str(x))
)

group_member_map = defaultdict(list)
for _, row in group_edges.drop_duplicates("group_idx").dropna(
    subset=["group_idx", "item_idx"]
).iterrows():
    gidx = int(row["group_idx"])
    iidx = int(row["item_idx"])
    members = item_to_users.get(iidx, [])[:MAX_GROUP_SIZE]
    group_member_map[gidx] = [int(m) for m in members if pd.notna(m)]

print(f"  Groups with members: {len(group_member_map)}")

# ── Train/Test Split (80/20 by timestamp — paper Section VI-C) ────────────
# Since we don't have timestamps in group_edges export, use random 80/20 split
print("\nSplitting group-item data into train/test (80/20)...")
np.random.seed(42)
n_groups = len(group_edges)
test_mask = np.random.rand(n_groups) > 0.8
test_set = group_edges[test_mask].dropna(subset=["group_idx", "item_idx"])
train_set = group_edges[~test_mask].dropna(subset=["group_idx", "item_idx"])

print(f"  Train: {len(train_set)} | Test: {len(test_set)}")

# ── Build set of items each group has interacted with (for exclusion) ──────
train_group_items = train_set.groupby("group_idx")["item_idx"].apply(set).to_dict()

# ── Scoring functions ──────────────────────────────────────────────────────
def get_group_embedding(member_indices):
    """Compute group embedding using attention-weighted member embeddings.
    Implements Equations 11 + 12 from the paper."""
    if len(member_indices) == 0:
        return np.zeros(user_emb.shape[1])
    
    # Get member embeddings
    u_embs = user_emb[member_indices]  # (G, D)
    
    # Get social influences and compute softmax (Eq. 12)
    gamma = social_influence[member_indices]  # (G,)
    exp_gamma = np.exp(gamma - np.max(gamma))  # numerical stability
    weights = exp_gamma / exp_gamma.sum()  # (G,)
    
    # Weighted sum (Eq. 11)
    group_emb = (u_embs * weights[:, np.newaxis]).sum(axis=0)  # (D,)
    return group_emb

def score_all_items(group_emb):
    """Score all items for a group. Returns array of shape (N_ITEMS,)."""
    return item_emb @ group_emb  # dot product

# ── Evaluation (Paper Section VI-C) ───────────────────────────────────────
print("\nEvaluating group recommendations...")

hits = {n: 0 for n in TOP_N_VALUES}
reciprocal_ranks = []
total_test = 0

for idx, (_, row) in enumerate(test_set.iterrows()):
    gidx = int(row["group_idx"])
    true_item = int(row["item_idx"])
    
    # Get group members
    members = group_member_map.get(gidx, [])
    if len(members) == 0:
        continue
    
    # Compute group embedding
    group_emb = get_group_embedding(members)
    
    # Score all items
    scores = score_all_items(group_emb)
    
    # Exclude items the group has already interacted with in training
    # (except the true item we're testing)
    excluded = train_group_items.get(gidx, set())
    for exc_item in excluded:
        if int(exc_item) != true_item:
            scores[int(exc_item)] = -np.inf
    
    # Rank items
    ranked_items = np.argsort(scores)[::-1]
    
    # Find rank of true item
    rank = np.where(ranked_items == true_item)[0]
    if len(rank) == 0:
        continue
    rank = rank[0] + 1  # 1-indexed
    
    # Compute Hits@n (Eq. 27)
    for n in TOP_N_VALUES:
        if rank <= n:
            hits[n] += 1
    
    # Compute reciprocal rank for MRR (Eq. 28)
    reciprocal_ranks.append(1.0 / rank)
    total_test += 1
    
    if (idx + 1) % 1000 == 0:
        print(f"  Evaluated {idx+1}/{len(test_set)} test cases...")

# ── Results ────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("SIGR MODEL EVALUATION RESULTS")
print("=" * 60)
print(f"\nTest cases evaluated: {total_test}")
print()

for n in TOP_N_VALUES:
    hits_at_n = hits[n] / total_test if total_test > 0 else 0
    print(f"  Hits@{n:>2}: {hits_at_n:.4f}")

mrr = np.mean(reciprocal_ranks) if reciprocal_ranks else 0
print(f"\n  MRR:     {mrr:.4f}")

print("\n" + "=" * 60)
print("Paper reference values (Yelp dataset):")
print("  SIGR — Hits@5: ~0.15 | Hits@10: ~0.22 | MRR: ~0.11")
print("=" * 60)

# ── User-level recommendation evaluation ──────────────────────────────────
print("\n\nEvaluating individual user recommendations...")

# Random sample of 5000 users for efficiency
sample_users = np.random.choice(
    interactions["user_idx"].dropna().unique().astype(int),
    size=min(5000, len(interactions["user_idx"].dropna().unique())),
    replace=False
)

user_hits = {n: 0 for n in TOP_N_VALUES}
user_rrs = []
user_total = 0

# Build user-item interaction set
user_item_set = interactions.dropna(subset=["user_idx", "item_idx"]).groupby(
    "user_idx"
)["item_idx"].apply(set).to_dict()

for uidx in sample_users:
    uidx = int(uidx)
    items = user_item_set.get(uidx, set())
    if len(items) < 2:
        continue
    
    items_list = list(items)
    # Hold out last item as test
    test_item = items_list[-1]
    train_items = set(items_list[:-1])
    
    # Score all items
    scores = item_emb @ user_emb[uidx]
    
    # Exclude training items
    for ti in train_items:
        scores[int(ti)] = -np.inf
    
    ranked = np.argsort(scores)[::-1]
    rank = np.where(ranked == int(test_item))[0]
    if len(rank) == 0:
        continue
    rank = rank[0] + 1
    
    for n in TOP_N_VALUES:
        if rank <= n:
            user_hits[n] += 1
    user_rrs.append(1.0 / rank)
    user_total += 1

print(f"\nUser test cases: {user_total}")
for n in TOP_N_VALUES:
    h = user_hits[n] / user_total if user_total > 0 else 0
    print(f"  Hits@{n:>2}: {h:.4f}")
user_mrr = np.mean(user_rrs) if user_rrs else 0
print(f"\n  MRR:     {user_mrr:.4f}")

print("\nPaper reference (BGEM individual):")
print("  Hits@5: ~0.25 | Hits@10: ~0.35 | MRR: ~0.17")
