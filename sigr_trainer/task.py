import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
import pyarrow.parquet as pq
import pyarrow.fs as pafs
from collections import defaultdict
import gc
import json
import io

BUCKET           = "yelp-sigr-training"
DATA_PREFIX      = "sigr-training"
EMBEDDING_DIM    = 50
NEGATIVE_SAMPLES = 6
LEARNING_RATE    = 0.001
EPOCHS           = 10
BATCH_SIZE       = 512
ETA              = 0.5
MAX_GROUP_SIZE   = 10

sys.stdout.reconfigure(line_buffering=True)

def load_parquet(blob_prefix, columns=None):
    fs = pafs.GcsFileSystem()
    dataset = pq.ParquetDataset(
        f"{BUCKET}/{DATA_PREFIX}/{blob_prefix}",
        filesystem=fs
    )
    table = dataset.read(columns=columns)
    return table.to_pandas()

# ── Load data ─────────────────────────────────────────────────────────────
print("Loading data...")
interactions = load_parquet("user_business_interactions", columns=[
    "user_id", "business_id", "review_id", "bgem_edge_weight"
])
print(f"  interactions: {len(interactions)}")

group_edges = load_parquet("group_item_edges", columns=[
    "group_id", "item_id", "edge_weight", "visit_dt"
])
print(f"  group_edges: {len(group_edges)}")

# Load implicit_groups — this has the ACTUAL group membership
# (friends who co-visited same business at same time)
implicit_groups = load_parquet("implicit_groups", columns=[
    "group_id", "user_id", "business_id"
])
print(f"  implicit_groups: {len(implicit_groups)}")
print("Data loaded.")

# ── ID Encoding ───────────────────────────────────────────────────────────
# Include users from both interactions AND implicit_groups
all_users = pd.concat([
    interactions["user_id"],
    implicit_groups["user_id"]
]).dropna().unique()

all_items = pd.concat([
    interactions["business_id"],
    group_edges["item_id"]
]).dropna().unique()

all_groups = group_edges["group_id"].dropna().unique()

user2idx  = {u: i for i, u in enumerate(all_users)}
item2idx  = {v: i for i, v in enumerate(all_items)}
group2idx = {g: i for i, g in enumerate(all_groups)}

N_USERS  = len(user2idx)
N_ITEMS  = len(item2idx)
N_GROUPS = len(group2idx)

print(f"Users: {N_USERS} | Items: {N_ITEMS} | Groups: {N_GROUPS}")

# ── Build group member map from implicit_groups (Paper Section VI-B) ──────
# This is the correct approach: group members are friends who co-visited
print("Building group member map from implicit_groups...")
implicit_groups["user_idx"] = implicit_groups["user_id"].map(user2idx)
implicit_groups["group_idx"] = implicit_groups["group_id"].map(group2idx)

group_member_map = defaultdict(list)
for gidx, members_df in implicit_groups.dropna(
    subset=["group_idx", "user_idx"]
).groupby("group_idx"):
    member_list = members_df["user_idx"].astype(int).unique().tolist()
    group_member_map[int(gidx)] = member_list[:MAX_GROUP_SIZE]

print(f"  Groups with members: {len(group_member_map)}")
avg_size = np.mean([len(m) for m in group_member_map.values()])
print(f"  Average group size: {avg_size:.2f}")

# Free implicit_groups memory
del implicit_groups
gc.collect()

# ── Build user-item map for negative sampling ─────────────────────────────
print("Building user-item map...")
interactions["user_idx"] = interactions["user_id"].map(user2idx)
interactions["item_idx"] = interactions["business_id"].map(item2idx)

uv_member_map = interactions.dropna(
    subset=["user_idx", "item_idx"]
).groupby("user_idx")["item_idx"].apply(
    lambda x: x.astype(int).tolist()
).to_dict()
print("User-item map built.")

# ── Train/test split by timestamp (Paper Section VI-C) ────────────────────
# "use the 80th percentile as the cut-off point"
print("Splitting group-item data by timestamp (80/20)...")
group_edges["group_idx"] = group_edges["group_id"].map(group2idx)
group_edges["item_idx"] = group_edges["item_id"].map(item2idx)

if "visit_dt" in group_edges.columns:
    group_edges = group_edges.sort_values("visit_dt")
    cutoff_idx = int(len(group_edges) * 0.8)
    train_gv = group_edges.iloc[:cutoff_idx].dropna(subset=["group_idx", "item_idx"])
    test_gv = group_edges.iloc[cutoff_idx:].dropna(subset=["group_idx", "item_idx"])
else:
    # Fallback to random split if no timestamp
    np.random.seed(42)
    mask = np.random.rand(len(group_edges)) > 0.8
    train_gv = group_edges[~mask].dropna(subset=["group_idx", "item_idx"])
    test_gv = group_edges[mask].dropna(subset=["group_idx", "item_idx"])

print(f"  Train G_GV: {len(train_gv)} | Test G_GV: {len(test_gv)}")

# ── Prepare arrays ────────────────────────────────────────────────────────
uv_users   = interactions["user_idx"].fillna(0).astype(int).values
uv_items   = interactions["item_idx"].fillna(0).astype(int).values
uv_weights = interactions["bgem_edge_weight"].values.astype(np.float32)

# Use only TRAINING group edges for training
gv_groups  = train_gv["group_idx"].fillna(0).astype(int).values
gv_items   = train_gv["item_idx"].fillna(0).astype(int).values
gv_weights = train_gv["edge_weight"].values.astype(np.float32)

# Save test set to GCS for evaluation
print("Saving test set to GCS...")
from google.cloud import storage
gcs_client = storage.Client()
gcs_bucket = gcs_client.bucket(BUCKET)

test_records = test_gv[["group_idx", "item_idx"]].to_json(orient="records")
blob = gcs_bucket.blob("models/sigr_v1/test_set.json")
blob.upload_from_string(test_records)
print(f"  Saved test set: {len(test_gv)} records")

del interactions, group_edges, train_gv, test_gv
gc.collect()

print(f"G_UV: {len(uv_users)} | G_GV (train): {len(gv_groups)}")

# ── Padding helper ────────────────────────────────────────────────────────
def get_member_arrays(group_indices):
    batch_members = []
    batch_masks   = []
    for gidx in group_indices:
        members = group_member_map.get(int(gidx), [0])[:MAX_GROUP_SIZE]
        if len(members) == 0:
            members = [0]
        pad_len = MAX_GROUP_SIZE - len(members)
        padded  = members + [0] * pad_len
        mask    = [1.0] * len(members) + [0.0] * pad_len
        batch_members.append(padded)
        batch_masks.append(mask)
    return (
        tf.constant(batch_members, dtype=tf.int32),
        tf.constant(batch_masks,   dtype=tf.float32)
    )

# ── BGEM Model ────────────────────────────────────────────────────────────
class BGEMModel(tf.keras.Model):
    def __init__(self, n_users, n_items, n_groups, emb_dim):
        super().__init__()
        self.user_emb = tf.keras.layers.Embedding(
            n_users, emb_dim,
            embeddings_initializer="glorot_uniform",
            name="user_embedding"
        )
        self.item_emb = tf.keras.layers.Embedding(
            n_items, emb_dim,
            embeddings_initializer="glorot_uniform",
            name="item_embedding"
        )
        self.social_influence = tf.Variable(
            tf.zeros([n_users]),
            trainable=True,
            name="social_influence"
        )

    def get_group_emb(self, member_ids, member_mask):
        u_embs     = self.user_emb(member_ids)
        influences = tf.gather(self.social_influence, member_ids)
        influences = influences + (1.0 - member_mask) * (-1e9)
        weights    = tf.nn.softmax(influences, axis=-1)
        weights    = weights * member_mask
        weights    = tf.expand_dims(weights, -1)
        return tf.reduce_sum(u_embs * weights, axis=1)

    def score_user_item(self, user_ids, item_ids):
        u = self.user_emb(user_ids)
        v = self.item_emb(item_ids)
        return tf.reduce_sum(u * v, axis=-1)

    def score_group_item(self, member_ids, member_mask, item_ids):
        g = self.get_group_emb(member_ids, member_mask)
        v = self.item_emb(item_ids)
        return tf.reduce_sum(g * v, axis=-1)

# ── Negative Sampling ─────────────────────────────────────────────────────
item_popularity = (
    pd.DataFrame({"item_idx": uv_items})
    .groupby("item_idx")
    .size()
    .reset_index(name="count")
)
item_popularity["prob"] = item_popularity["count"] ** 0.75
item_popularity["prob"] /= item_popularity["prob"].sum()

item_pop_indices = item_popularity["item_idx"].values
item_pop_probs   = item_popularity["prob"].values

def sample_negatives_uv(n, k):
    return np.random.choice(item_pop_indices, size=(n, k), p=item_pop_probs)

def sample_negatives_gv(group_indices, k):
    batch_negs = []
    for gidx in group_indices:
        members = group_member_map.get(int(gidx), [])
        if len(members) == 0:
            negs = np.random.choice(item_pop_indices, size=k, p=item_pop_probs)
        else:
            items = [i for m in members for i in uv_member_map.get(m, [])]
            if len(items) == 0:
                negs = np.random.choice(item_pop_indices, size=k, p=item_pop_probs)
            else:
                counts  = np.bincount(items, minlength=N_ITEMS).astype(float)
                counts  = (counts + 1e-6) ** 0.75
                counts /= counts.sum()
                negs    = np.random.choice(N_ITEMS, size=k, p=counts)
        batch_negs.append(negs)
    return np.array(batch_negs)

# ── BPR Loss ──────────────────────────────────────────────────────────────
def bpr_loss(pos_scores, neg_scores, edge_weights):
    pos_loss = tf.math.log_sigmoid(pos_scores)
    neg_loss = tf.reduce_mean(
        tf.math.log_sigmoid(-neg_scores), axis=-1
    )
    return -tf.reduce_mean(edge_weights * (pos_loss + neg_loss))

# ── Model + Optimiser ─────────────────────────────────────────────────────
model     = BGEMModel(N_USERS, N_ITEMS, N_GROUPS, EMBEDDING_DIM)
optimizer = tf.keras.optimizers.Adam(LEARNING_RATE)

@tf.function
def train_step_uv(u_ids, v_pos_ids, v_neg_ids, weights):
    with tf.GradientTape() as tape:
        pos_scores = model.score_user_item(u_ids, v_pos_ids)
        neg_scores = tf.stack([
            model.score_user_item(u_ids, v_neg_ids[:, k])
            for k in range(NEGATIVE_SAMPLES)
        ], axis=1)
        loss = bpr_loss(pos_scores, neg_scores, weights)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss

@tf.function
def train_step_gv(member_ids, member_mask, v_pos_ids, v_neg_ids, weights):
    with tf.GradientTape() as tape:
        pos_scores = model.score_group_item(member_ids, member_mask, v_pos_ids)
        neg_scores = tf.stack([
            model.score_group_item(member_ids, member_mask, v_neg_ids[:, k])
            for k in range(NEGATIVE_SAMPLES)
        ], axis=1)
        loss = bpr_loss(pos_scores, neg_scores, weights)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss

# ── Joint Training Loop (Algorithm 1) ─────────────────────────────────────
print("Starting joint training...")
n_uv = len(uv_users)
n_gv = len(gv_groups)

for epoch in range(EPOCHS):
    uv_idx = np.random.permutation(n_uv)
    gv_idx = np.random.permutation(n_gv)

    uv_u = uv_users[uv_idx]
    uv_v = uv_items[uv_idx]
    uv_w = uv_weights[uv_idx]
    gv_g = gv_groups[gv_idx]
    gv_v = gv_items[gv_idx]
    gv_w = gv_weights[gv_idx]

    epoch_loss_uv = 0.0
    epoch_loss_gv = 0.0
    n_uv_batches  = 0
    n_gv_batches  = 0
    uv_start = 0
    gv_start = 0
    total_steps = (n_uv + n_gv) // BATCH_SIZE

    for step in range(total_steps):
        use_gv = np.random.rand() < (1.0 / (1.0 + ETA))

        if use_gv and gv_start < n_gv:
            end     = min(gv_start + BATCH_SIZE, n_gv)
            g_batch = gv_g[gv_start:end]
            v_batch = tf.constant(gv_v[gv_start:end], dtype=tf.int32)
            w_batch = tf.constant(gv_w[gv_start:end], dtype=tf.float32)
            member_ids, member_mask = get_member_arrays(g_batch)
            neg_batch = tf.constant(
                sample_negatives_gv(g_batch, NEGATIVE_SAMPLES),
                dtype=tf.int32
            )
            loss = train_step_gv(
                member_ids, member_mask, v_batch, neg_batch, w_batch
            )
            epoch_loss_gv += loss.numpy()
            n_gv_batches  += 1
            gv_start       = end
        else:
            if uv_start >= n_uv:
                continue
            end     = min(uv_start + BATCH_SIZE, n_uv)
            u_batch = tf.constant(uv_u[uv_start:end], dtype=tf.int32)
            v_batch = tf.constant(uv_v[uv_start:end], dtype=tf.int32)
            w_batch = tf.constant(uv_w[uv_start:end], dtype=tf.float32)
            neg_batch = tf.constant(
                sample_negatives_uv(end - uv_start, NEGATIVE_SAMPLES),
                dtype=tf.int32
            )
            loss = train_step_uv(u_batch, v_batch, neg_batch, w_batch)
            epoch_loss_uv += loss.numpy()
            n_uv_batches  += 1
            uv_start       = end

        if step % 1000 == 0:
            print(f"  step {step}/{total_steps}")

    print(f"Epoch {epoch+1}/{EPOCHS} — "
          f"G_UV Loss: {epoch_loss_uv/max(n_uv_batches,1):.4f} | "
          f"G_GV Loss: {epoch_loss_gv/max(n_gv_batches,1):.4f}")

# ── Save model artifacts ──────────────────────────────────────────────────
print("Saving model artifacts...")

def save_numpy_to_gcs(array, blob_name):
    buf = io.BytesIO()
    np.save(buf, array)
    buf.seek(0)
    blob = gcs_bucket.blob(blob_name)
    blob.upload_from_file(buf)
    print(f"  Saved {blob_name} — shape {array.shape}")

save_numpy_to_gcs(model.user_emb.embeddings.numpy(), "models/sigr_v1/user_embeddings.npy")
save_numpy_to_gcs(model.item_emb.embeddings.numpy(), "models/sigr_v1/item_embeddings.npy")
save_numpy_to_gcs(model.social_influence.numpy(), "models/sigr_v1/social_influence.npy")

# Save ID mappings
mappings = {
    "user2idx": {str(k): int(v) for k, v in user2idx.items()},
    "item2idx": {str(k): int(v) for k, v in item2idx.items()},
    "group2idx": {str(k): int(v) for k, v in group2idx.items()},
}
for name, mapping in mappings.items():
    blob = gcs_bucket.blob(f"models/sigr_v1/{name}.json")
    blob.upload_from_string(json.dumps(mapping))
    print(f"  Saved {name}.json — {len(mapping)} entries")

# Save group member map for evaluation
gm_serializable = {str(k): v for k, v in group_member_map.items()}
blob = gcs_bucket.blob("models/sigr_v1/group_member_map.json")
blob.upload_from_string(json.dumps(gm_serializable))
print(f"  Saved group_member_map.json — {len(gm_serializable)} groups")

print("Model saved successfully!")
