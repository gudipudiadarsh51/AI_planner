"""
SIGR Recommendation API
Serves group and individual restaurant recommendations
with dietary preference and location filtering.

Endpoints:
    GET  /health              — Health check
    POST /recommend/user      — Recommend items for a single user
    POST /recommend/group     — Recommend items for a group of users
    GET  /business/{id}       — Get business details
"""

import os
import io
import json
import numpy as np
from collections import defaultdict
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from google.cloud import storage
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────
BUCKET       = os.environ.get("GCS_BUCKET", "yelp-sigr-training")
MODEL_PREFIX = os.environ.get("MODEL_PREFIX", "models/sigr_v1")
BIZ_PREFIX   = os.environ.get("BIZ_PREFIX", "sigr-training/business_features")

app = FastAPI(
    title="SIGR Group Restaurant Recommender",
    description="Social Influence-based Group Recommendation API for Yelp restaurants",
    version="1.0.0"
)

# ── Global model state (loaded once at startup) ───────────────────────────
model = {}


# ── Request/Response models ───────────────────────────────────────────────
class UserRecommendRequest(BaseModel):
    user_id: str
    top_n: int = 10
    filters: Optional[dict] = None
    # Example filters:
    # {
    #   "halal": true,
    #   "vegetarian": true,
    #   "city": "Los Angeles",
    #   "min_stars": 3.5,
    #   "max_distance_km": 10,
    #   "latitude": 34.0522,
    #   "longitude": -118.2437
    # }


class GroupRecommendRequest(BaseModel):
    user_ids: List[str]
    top_n: int = 10
    filters: Optional[dict] = None


class RecommendedItem(BaseModel):
    business_id: str
    name: str
    score: float
    stars: Optional[float] = None
    city: Optional[str] = None
    categories: Optional[str] = None
    tag_halal: Optional[bool] = None
    tag_south_asian: Optional[bool] = None
    tag_vegetarian: Optional[bool] = None
    tag_gluten_free: Optional[bool] = None
    distance_km: Optional[float] = None


class RecommendResponse(BaseModel):
    recommendations: List[RecommendedItem]
    group_size: int
    member_influences: Optional[dict] = None


# ── Model loading ─────────────────────────────────────────────────────────
def load_numpy_from_gcs(gcs_bucket, blob_name):
    blob = gcs_bucket.blob(blob_name)
    buf = io.BytesIO()
    blob.download_to_file(buf)
    buf.seek(0)
    return np.load(buf)


def load_json_from_gcs(gcs_bucket, blob_name):
    blob = gcs_bucket.blob(blob_name)
    return json.loads(blob.download_as_text())


@app.on_event("startup")
async def load_model():
    """Load all model artifacts into memory at startup."""
    logger.info("Loading model artifacts from GCS...")
    client = storage.Client()
    gcs_bucket = client.bucket(BUCKET)

    # Load embeddings
    model["user_emb"] = load_numpy_from_gcs(
        gcs_bucket, f"{MODEL_PREFIX}/user_embeddings.npy"
    )
    model["item_emb"] = load_numpy_from_gcs(
        gcs_bucket, f"{MODEL_PREFIX}/item_embeddings.npy"
    )
    model["social_influence"] = load_numpy_from_gcs(
        gcs_bucket, f"{MODEL_PREFIX}/social_influence.npy"
    )

    # Load ID mappings
    model["user2idx"] = load_json_from_gcs(
        gcs_bucket, f"{MODEL_PREFIX}/user2idx.json"
    )
    model["item2idx"] = load_json_from_gcs(
        gcs_bucket, f"{MODEL_PREFIX}/item2idx.json"
    )
    model["idx2item"] = {int(v): k for k, v in model["item2idx"].items()}

    logger.info(f"  User embeddings: {model['user_emb'].shape}")
    logger.info(f"  Item embeddings: {model['item_emb'].shape}")
    logger.info(f"  Social influence: {model['social_influence'].shape}")

    # Load business metadata for filtering
    logger.info("Loading business metadata...")
    try:
        import pyarrow.parquet as pq
        import pyarrow.fs as pafs
        fs = pafs.GcsFileSystem()
        dataset = pq.ParquetDataset(
            f"{BUCKET}/{BIZ_PREFIX}", filesystem=fs
        )
        biz_df = dataset.read().to_pandas()
        model["business_meta"] = biz_df.set_index("business_id").to_dict("index")
        logger.info(f"  Business metadata: {len(model['business_meta'])} businesses")
    except Exception as e:
        logger.warning(f"  Could not load business metadata: {e}")
        model["business_meta"] = {}

    logger.info("Model loaded successfully.")


# ── Core recommendation logic ─────────────────────────────────────────────
def get_group_embedding(user_indices: List[int]) -> np.ndarray:
    """Compute group embedding using attention-weighted member embeddings.
    Implements Equations 11 + 12 from the SIGR paper."""
    if len(user_indices) == 0:
        raise ValueError("Empty user list")

    u_embs = model["user_emb"][user_indices]          # (G, D)
    gamma = model["social_influence"][user_indices]    # (G,)

    # Softmax with numerical stability (Eq. 12)
    exp_gamma = np.exp(gamma - np.max(gamma))
    weights = exp_gamma / exp_gamma.sum()              # (G,)

    # Weighted sum of member embeddings (Eq. 11)
    group_emb = (u_embs * weights[:, np.newaxis]).sum(axis=0)
    return group_emb, weights


def score_items(query_emb: np.ndarray) -> np.ndarray:
    """Score all items against a query embedding (user or group)."""
    return model["item_emb"] @ query_emb


def haversine_km(lat1, lon1, lat2, lon2):
    """Calculate distance between two points in kilometers."""
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


def apply_filters(scores: np.ndarray, filters: dict) -> np.ndarray:
    """Apply dietary, location, and rating filters to item scores.
    Filtered-out items get -inf score."""
    if not filters or not model.get("business_meta"):
        return scores

    filtered_scores = scores.copy()

    for item_idx in range(len(scores)):
        biz_id = model["idx2item"].get(item_idx)
        if biz_id is None:
            continue
        meta = model["business_meta"].get(biz_id, {})
        if not meta:
            continue

        # Dietary filters
        if filters.get("halal") and not meta.get("tag_halal", False):
            filtered_scores[item_idx] = -np.inf
            continue

        if filters.get("south_asian") and not meta.get("tag_south_asian", False):
            filtered_scores[item_idx] = -np.inf
            continue

        if filters.get("vegetarian") and not meta.get("tag_vegetarian", False):
            filtered_scores[item_idx] = -np.inf
            continue

        if filters.get("gluten_free") and not meta.get("tag_gluten_free", False):
            filtered_scores[item_idx] = -np.inf
            continue

        # Minimum stars filter
        min_stars = filters.get("min_stars")
        if min_stars and meta.get("avg_stars_yelp", 0) < min_stars:
            filtered_scores[item_idx] = -np.inf
            continue

        # City filter
        city = filters.get("city")
        if city and meta.get("city", "").lower() != city.lower():
            filtered_scores[item_idx] = -np.inf
            continue

        # Distance filter
        max_dist = filters.get("max_distance_km")
        user_lat = filters.get("latitude")
        user_lon = filters.get("longitude")
        if max_dist and user_lat and user_lon:
            biz_lat = meta.get("latitude")
            biz_lon = meta.get("longitude")
            if biz_lat and biz_lon:
                dist = haversine_km(user_lat, user_lon, biz_lat, biz_lon)
                if dist > max_dist:
                    filtered_scores[item_idx] = -np.inf

    return filtered_scores


def build_recommendations(
    scores: np.ndarray, top_n: int, user_lat=None, user_lon=None
) -> List[RecommendedItem]:
    """Build ranked recommendation list from scores."""
    top_indices = np.argsort(scores)[::-1][:top_n]
    recommendations = []

    for item_idx in top_indices:
        if scores[item_idx] == -np.inf:
            break

        biz_id = model["idx2item"].get(int(item_idx), "unknown")
        meta = model.get("business_meta", {}).get(biz_id, {})

        # Compute distance if user location provided
        distance = None
        if user_lat and user_lon and meta.get("latitude") and meta.get("longitude"):
            distance = round(
                haversine_km(user_lat, user_lon, meta["latitude"], meta["longitude"]),
                2
            )

        recommendations.append(RecommendedItem(
            business_id=biz_id,
            name=meta.get("name", "Unknown"),
            score=round(float(scores[item_idx]), 4),
            stars=meta.get("avg_stars_yelp"),
            city=meta.get("city"),
            categories=meta.get("categories"),
            tag_halal=meta.get("tag_halal"),
            tag_south_asian=meta.get("tag_south_asian"),
            tag_vegetarian=meta.get("tag_vegetarian"),
            tag_gluten_free=meta.get("tag_gluten_free"),
            distance_km=distance,
        ))

    return recommendations


# ── API Endpoints ──────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "model_loaded": "user_emb" in model,
        "n_users": model.get("user_emb", np.array([])).shape[0],
        "n_items": model.get("item_emb", np.array([])).shape[0],
    }


@app.post("/recommend/user", response_model=RecommendResponse)
async def recommend_for_user(request: UserRecommendRequest):
    """Recommend restaurants for a single user.
    Uses BGEM user-item dot product (Eq. 1)."""

    user_idx = model["user2idx"].get(request.user_id)
    if user_idx is None:
        raise HTTPException(status_code=404, detail=f"User {request.user_id} not found")
    user_idx = int(user_idx)

    # Score all items
    query_emb = model["user_emb"][user_idx]
    scores = score_items(query_emb)

    # Apply filters
    filters = request.filters or {}
    scores = apply_filters(scores, filters)

    # Build response
    recs = build_recommendations(
        scores, request.top_n,
        user_lat=filters.get("latitude"),
        user_lon=filters.get("longitude")
    )

    return RecommendResponse(
        recommendations=recs,
        group_size=1,
        member_influences={request.user_id: 1.0}
    )


@app.post("/recommend/group", response_model=RecommendResponse)
async def recommend_for_group(request: GroupRecommendRequest):
    """Recommend restaurants for a group of users.
    Uses SIGR attention-weighted group embedding (Eqs. 11 + 12)."""

    # Resolve user IDs to indices
    user_indices = []
    valid_ids = []
    for uid in request.user_ids:
        idx = model["user2idx"].get(uid)
        if idx is not None:
            user_indices.append(int(idx))
            valid_ids.append(uid)

    if len(user_indices) == 0:
        raise HTTPException(
            status_code=404,
            detail="None of the provided user IDs were found in the model"
        )

    # Compute group embedding with social influence weights
    group_emb, weights = get_group_embedding(user_indices)

    # Score all items
    scores = score_items(group_emb)

    # Apply filters
    filters = request.filters or {}
    scores = apply_filters(scores, filters)

    # Build response
    recs = build_recommendations(
        scores, request.top_n,
        user_lat=filters.get("latitude"),
        user_lon=filters.get("longitude")
    )

    # Include member influence weights in response
    member_influences = {
        uid: round(float(w), 4) for uid, w in zip(valid_ids, weights)
    }

    return RecommendResponse(
        recommendations=recs,
        group_size=len(user_indices),
        member_influences=member_influences
    )


@app.get("/business/{business_id}")
async def get_business(business_id: str):
    """Get business details and tags."""
    meta = model.get("business_meta", {}).get(business_id)
    if meta is None:
        raise HTTPException(status_code=404, detail=f"Business {business_id} not found")
    return meta


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))