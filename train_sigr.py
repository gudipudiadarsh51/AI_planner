import os
import numpy as np
import pandas as pd
import tensorflow as tf
import pyarrow.parquet as pq
import pyarrow.fs as pafs
from collections import defaultdict

BUCKET           = "yelp-sigr-training"
DATA_PREFIX      = "sigr-training"
EMBEDDING_DIM    = 50
NEGATIVE_SAMPLES = 6
LEARNING_RATE    = 0.001
EPOCHS           = 10
BATCH_SIZE       = 512
ETA              = 0.5
MAX_GROUP_SIZE   = 10

def load_parquet(blob_prefix):
    fs = pafs.GcsFileSystem()
    dataset = pq.ParquetDataset(
        f"{BUCKET}/{DATA_PREFIX}/{blob_prefix}",
        filesystem=fs
    )
    return dataset.read_pandas().to_pandas()

print("Loading data...")
interactions = load_parquet("user_business_interactions")
social_edges = load_parquet("user_social_edges")
group_edges  = load_parquet("group_item_edges")
print(f"Loaded: interactions={len(interactions)} "
      f"social_edges={len(social_edges)} "
      f"group_edges={len(group_edges)}")

# ── ID Encoding ───────────────────────────────────────────────────────────
all_users = pd.concat([
    interactions["user_id"],
    social_edges["src_user_id"],
    social_edges["dst_user_id"]
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

# ── Vectorised group member map ───────────────────────────────────────────
# No iterrows — use groupby on group_edges directly
# Each group_id maps to the users who interacted with it via reviews
print("Building group member map...")
interactions["user_idx"]  = interactions["user_id"].map(user2idx)
interactions["item_idx"]  = interactions["business_id"].map(item2idx)
group_edges["group_idx"]  = group_edges["group_id"].map(group2idx)
group_edges["item_idx"]   = group_edges["item_id"].map(item2idx)

# Build group→members by joining group_edges items back to interactions
ge_items = group_edges[["group_idx", "item_idx"]].dropna()
ge_items["item_idx"] = ge_items["item_idx"].astype(int)

# For each group, find users who reviewed any item in that group
item_to_users = interactions.groupby("item_idx")["user_idx"].apply(list).to_dict()

group_member_map = defaultdict(list)
for _, row in ge_items.drop_duplicates("group_idx").iterrows():
    gidx = int(row["group_idx"])
    iidx = int(row["item_idx"])
    members = item_to_users.get(iidx, [])[:MAX_GROUP_SIZE]
    group_member_map[gidx] = [int(m) for m in members if pd.notna(m)]

print(f"Group member map built: {len(group_member_map)} groups with members")

# ── Vectorised user→items map for negative sampling ───────────────────────
uv_member_map = interactions.groupby("user_idx")["item_idx"].apply(list).to_dict()
print("User-item map built.")

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
    interactions
    .groupby("business_id")["review_id"]
    .count()
    .reset_index()
    .rename(columns={"review_id": "count"})
)
item_popularity["prob"] = item_popularity["count"] ** 0.75
item_popularity["prob"] /= item_popularity["prob"].sum()

item_pop_indices = item_popularity["business_id"].map(
    item2idx).fillna(0).astype(int).values
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
                counts   = (counts + 1e-6) ** 0.75
                counts  /= counts.sum()
                negs     = np.random.choice(N_ITEMS, size=k, p=counts)
        batch_negs.append(negs)
    return np.array(batch_negs)

# ── BPR Loss ──────────────────────────────────────────────────────────────
def bpr_loss(pos_scores, neg_scores, edge_weights):
    pos_loss = tf.math.log_sigmoid(pos_scores)
    neg_loss = tf.reduce_mean(
        tf.math.log_sigmoid(-neg_scores), axis=-1
    )
    return -tf.reduce_mean(edge_weights * (pos_loss + neg_loss))

# ── Prepare arrays ────────────────────────────────────────────────────────
uv_users   = interactions["user_idx"].fillna(0).astype(int).values
uv_items   = interactions["item_idx"].fillna(0).astype(int).values
uv_weights = interactions["bgem_edge_weight"].values.astype(np.float32)

gv_groups  = group_edges["group_idx"].fillna(0).astype(int).values
gv_items   = group_edges["item_idx"].fillna(0).astype(int).values
gv_weights = group_edges["edge_weight"].values.astype(np.float32)

print(f"G_UV: {len(uv_users)} | G_GV: {len(gv_groups)}")

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

# ── Joint Training Loop ───────────────────────────────────────────────────
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

    for _ in range(total_steps):
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

    print(f"Epoch {epoch+1}/{EPOCHS} — "
          f"G_UV Loss: {epoch_loss_uv/max(n_uv_batches,1):.4f} | "
          f"G_GV Loss: {epoch_loss_gv/max(n_gv_batches,1):.4f}")

save_path = os.environ.get(
    "AIP_MODEL_DIR",
    "gs://yelp-sigr-training/models/sigr_v1"
)
model.save(save_path)
print(f"Model saved to {save_path}")