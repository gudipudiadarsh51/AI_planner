# mock_data.py
import numpy as np
import pandas as pd

SAMPLE_SIZE = 2000

def make_synthetic_data(n=SAMPLE_SIZE):
    user_ids     = [f"user_{i}" for i in range(200)]
    business_ids = [f"biz_{i}"  for i in range(300)]
    group_ids    = [f"grp_{i}"  for i in range(100)]

    interactions = pd.DataFrame({
        "user_id"           : np.random.choice(user_ids, n),
        "business_id"       : np.random.choice(business_ids, n),
        "review_id"         : [f"rev_{i}" for i in range(n)],
        "stars"             : np.random.choice([1,2,3,4,5], n).astype(float),
        "bgem_edge_weight"  : np.random.uniform(0.1, 1.0, n).astype(np.float32),
        "review_trust_score": np.random.uniform(0.5, 1.0, n).astype(np.float32),
    })

    social_edges = pd.DataFrame({
        "src_user_id": np.random.choice(user_ids, n),
        "dst_user_id": np.random.choice(user_ids, n),
    })

    group_edges = pd.DataFrame({
        "group_id"   : np.random.choice(group_ids, n),
        "item_id"    : np.random.choice(business_ids, n),
        "edge_weight": np.ones(n, dtype=np.float32),
    })

    # Group membership — 2 to 6 members per group (mirrors paper avg 4.45)
    group_members_rows = []
    for gid in group_ids:
        size = np.random.randint(2, 7)
        members = np.random.choice(user_ids, size, replace=False)
        for uid in members:
            group_members_rows.append({"group_id": gid, "user_id": uid})
    group_members = pd.DataFrame(group_members_rows)

    return interactions, social_edges, group_edges, group_members


if __name__ == "__main__":
    interactions, social_edges, group_edges, group_members = make_synthetic_data()
    print(f"interactions  : {interactions.shape}")
    print(f"social_edges  : {social_edges.shape}")
    print(f"group_edges   : {group_edges.shape}")
    print(f"group_members : {group_members.shape}")
    print(f"\nSample group members:\n{group_members.head(8)}")
    