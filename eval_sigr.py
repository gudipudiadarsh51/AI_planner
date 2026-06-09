"""
SIGR Model Evaluation — Paper-matched methodology
Uses temporal train/test split and proper group membership.
"""

import os
import io
import json
import numpy as np
import pandas as pd
from google.cloud import storage
from collections import defaultdict

BUCKET       = "yelp-sigr-training"
MODEL_PREFIX = "models/sigr_v1"
TOP_N_VALUES = [1, 5, 10, 15, 20]

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

user_emb         = load_numpy_from_gcs(f"{MODEL_PREFIX}/user_embeddings.npy")
item_emb         = load_numpy_from_gcs(f"{MODEL_PREFIX}/item_embeddings.npy")
social_influence = load_numpy_from_gcs(f"{MODEL_PREFIX}/social_influence.npy")
user2idx         = load_json_from_gcs(f"{MODEL_PREFIX}/user2idx.json")
item2idx         = load_json_from_gcs(f"{MODEL_PREFIX}/item2idx.json")
group2idx        = load_json_from_gcs(f"{MODEL_PREFIX}/group2idx.json")
group_member_map = load_json_from_gcs(f"{MODEL_PREFIX}/group_member_map.json")
test_set         = load_json_from_gcs(f"{MODEL_PREFIX}/test_set.json")

# Convert group_member_map keys to int
group_member_map = {int(k): v for k, v in group_member_map.items()}

idx2item = {int(v): k for k, v in item2idx.items()}
N_USERS = len(user2idx)
N_ITEMS = len(item2idx)

print(f"  User embeddings:    {user_emb.shape}")
print(f"  Item embeddings:    {item_emb.shape}")
print(f"  Social influence:   {social_influence.shape}")
print(f"  Group member map:   {len(group_member_map)} groups")
print(f"  Test set:           {len(test_set)} records")

# ── Build training group-item set for exclusion ───────────────────────────
# Load full group edges to build exclusion set
import pyarrow.parquet as pq
import pyarrow.fs as pafs
fs = pafs.GcsFileSystem()

group_edges = pq.ParquetDataset(
    f"{BUCKET}/sigr-training/group_item_edges", filesystem=fs
).read(columns=["group_id", "item_id", "visit_dt"]).to_pandas()

group_edges["group_idx"] = group_edges["group_id"].map(
    lambda x: group2idx.get(str(x))
)
group_edges["item_idx"] = group_edges["item_id"].map(
    lambda x: item2idx.get(str(x))
)

# Temporal split — training is first 80%
group_edges = group_edges.sort_values("visit_dt")
cutoff = int(len(group_edges) * 0.8)
train_edges = group_edges.iloc[:cutoff].dropna(subset=["group_idx", "item_idx"])

# Build exclusion set: items each group interacted with in training
train_group_items = train_edges.groupby("group_idx")["item_idx"].apply(
    lambda x: set(x.astype(int))
).to_dict()

del group_edges, train_edges
print(f"  Train exclusion sets built for {len(train_group_items)} groups")

# ── Scoring functions ──────────────────────────────────────────────────────
def get_group_embedding(member_indices):
    """Equations 11 + 12 from paper."""
    if len(member_indices) == 0:
        return np.zeros(user_emb.shape[1])
    u_embs = user_emb[member_indices]
    gamma = social_influence[member_indices]
    exp_gamma = np.exp(gamma - np.max(gamma))
    weights = exp_gamma / exp_gamma.sum()
    return (u_embs * weights[:, np.newaxis]).sum(axis=0)

def score_all_items(query_emb):
    return item_emb @ query_emb

# ── Group Recommendation Evaluation (Paper Section VI-C) ──────────────────
print("\nEvaluating group recommendations...")
print("(Using temporal split + proper group membership from implicit_groups)")

hits = {n: 0 for n in TOP_N_VALUES}
reciprocal_ranks = []
total_test = 0

for idx, record in enumerate(test_set):
    gidx = int(record["group_idx"])
    true_item = int(record["item_idx"])

    # Get ACTUAL group members (friends who co-visited)
    members = group_member_map.get(gidx, [])
    if len(members) == 0:
        continue

    # Compute group embedding
    group_emb = get_group_embedding(members)

    # Score all items
    scores = score_all_items(group_emb)

    # Exclude items the group interacted with in TRAINING only
    excluded = train_group_items.get(gidx, set())
    for exc_item in excluded:
        if int(exc_item) != true_item:
            scores[int(exc_item)] = -np.inf

    # Rank items
    ranked_items = np.argsort(scores)[::-1]

    # Find rank of true item (Eq. 27, 28)
    rank = np.where(ranked_items == true_item)[0]
    if len(rank) == 0:
        continue
    rank = rank[0] + 1

    for n in TOP_N_VALUES:
        if rank <= n:
            hits[n] += 1

    reciprocal_ranks.append(1.0 / rank)
    total_test += 1

    if (idx + 1) % 1000 == 0:
        print(f"  Evaluated {idx+1}/{len(test_set)} test cases...")

# ── Group Results ──────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("SIGR GROUP RECOMMENDATION RESULTS")
print("(Temporal split, proper group membership)")
print("=" * 60)
print(f"\nTest cases evaluated: {total_test}")
for n in TOP_N_VALUES:
    h = hits[n] / total_test if total_test > 0 else 0
    print(f"  Hits@{n:>2}: {h:.4f}")
mrr = np.mean(reciprocal_ranks) if reciprocal_ranks else 0
print(f"\n  MRR:     {mrr:.4f}")
print("\nPaper reference (SIGR on Yelp):")
print("  Hits@5: ~0.15 | Hits@10: ~0.22 | Hits@20: ~0.31 | MRR: ~0.11")

# ── Individual User Evaluation ─────────────────────────────────────────────
print("\n\nEvaluating individual user recommendations...")

interactions = pq.ParquetDataset(
    f"{BUCKET}/sigr-training/user_business_interactions", filesystem=fs
).read(columns=["user_id", "business_id"]).to_pandas()

interactions["user_idx"] = interactions["user_id"].map(
    lambda x: user2idx.get(str(x))
)
interactions["item_idx"] = interactions["business_id"].map(
    lambda x: item2idx.get(str(x))
)
interactions = interactions.dropna(subset=["user_idx", "item_idx"])

user_item_set = interactions.groupby("user_idx")["item_idx"].apply(
    lambda x: x.astype(int).tolist()
).to_dict()

sample_users = np.random.RandomState(42).choice(
    list(user_item_set.keys()),
    size=min(5000, len(user_item_set)),
    replace=False
)

user_hits = {n: 0 for n in TOP_N_VALUES}
user_rrs = []
user_total = 0

for uidx in sample_users:
    uidx = int(uidx)
    items = user_item_set.get(uidx, [])
    if len(items) < 2:
        continue

    test_item = items[-1]
    train_items = set(items[:-1])

    scores = item_emb @ user_emb[uidx]
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
print("\n" + "=" * 60)
