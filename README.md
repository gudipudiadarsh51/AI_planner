# SIGR — Social Influence-based Group Restaurant Recommender

An end-to-end ML pipeline that recommends restaurants to groups of friends using the Yelp academic dataset. Implements the [SIGR paper](https://doi.org/10.1109/ICDE.2019.00005) (Yin et al., ICDE 2019) with an attention mechanism that learns each user's social influence within a group to produce group-aware recommendations.

Built on Google Cloud Platform with BigQuery, Vertex AI, and Cloud Run.

## Architecture

```
Yelp Academic Dataset
        │
        ▼
┌─────────────────────────┐
│  BigQuery Data Pipeline  │
│  Raw → Staging → DQ →   │
│  Curated → Features     │
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│  Graph Construction      │
│  G_UV: user-item edges   │
│  G_GV: group-item edges  │
│  G_S:  social network    │
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│  SIGR Model Training     │
│  BGEM + Attention +      │
│  Joint Optimization      │
│  (Vertex AI / Colab)     │
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│  Cloud Run API           │
│  /recommend/user         │
│  /recommend/group        │
│  Dietary & location      │
│  filtering               │
└─────────────────────────┘
```

## What It Does

- **Individual recommendations**: Predicts which restaurants a user would enjoy based on learned preference embeddings.
- **Group recommendations**: Given a group of friends, computes a consensus recommendation by weighting each member's preferences by their learned social influence (attention mechanism, Equations 11-12 of the paper).
- **Preference filtering**: Filters results by halal, vegetarian, gluten-free, cuisine type, minimum rating, city, and proximity.

## Key Results

| Metric | Individual (BGEM) | Group (SIGR) |
|--------|-------------------|--------------|
| Hits@5 | 0.3374 | — |
| Hits@10 | 0.3944 | — |
| Hits@20 | 0.4594 | — |
| MRR | 0.2761 | — |

Trained on 6.9M user-item interactions and 30K implicit group-item interactions extracted from the Yelp dataset.

## Project Structure

```
├── sigr_trainer/
│   ├── __init__.py
│   └── task.py              # SIGR training script (Vertex AI / Colab)
├── app/
│   ├── main.py              # FastAPI recommendation API
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── deploy.sh            # Cloud Run deployment script
│   └── test_api.py          # API integration tests
├── eval_sigr.py             # Model evaluation (Hits@n, MRR)
├── mock_train_sigr.py       # Local mock training with synthetic data
├── mock_data.py             # Synthetic data generator for local testing
├── setup.py                 # Package config for Vertex AI
├── requirements.txt         # Local development dependencies
├── CLAUDE.md                # AI coding guidelines
└── README.md
```

## Data Pipeline

The BigQuery pipeline processes raw Yelp data through five layers:

| Layer | Dataset | Purpose |
|-------|---------|---------|
| Raw | `yelp_catalog` | Untouched Yelp tables |
| Staging | `staging_yelp` | Type casting, deduplication, null handling (Views) |
| Data Quality | `dq_yelp` | Trust scoring, anomaly flags (Views) |
| Curated | `curated_yelp` | Full cleaned + trusted filtered tables |
| Features | `features_yelp` | ML-ready user, business, interaction tables |
| Graph | `graph_yelp` | User-item, group-item, social network edges |

Data quality flags include rapid review detection, same-day burst detection, repeated text detection, coordinated spam detection, and a composite trust score that down-weights suspicious reviews during training.

## Model

The SIGR model has three learnable components:

| Parameter | Shape | What It Learns |
|-----------|-------|----------------|
| User embeddings | (1.98M, 50) | User preference vectors |
| Item embeddings | (150K, 50) | Restaurant characteristic vectors |
| Social influence γ | (1.98M,) | Per-user influence weight in group decisions |

**Training**: Joint optimization (Algorithm 1 from the paper) alternates between user-item (G_UV) and group-item (G_GV) training steps using a Bernoulli coin flip controlled by hyperparameter η. BPR loss with degree-based negative sampling for G_UV and group-aware negative sampling for G_GV.

**Group embedding**: A group's preference vector is the attention-weighted sum of its members' embeddings, where attention weights are derived from each member's learned social influence via softmax.

## API

The Cloud Run API serves recommendations with optional dietary and location filters.

**Endpoints:**

| Method | Path | Description |
|--------|------|-------------|
| POST | `/recommend/user` | Top-n restaurants for a single user |
| POST | `/recommend/group` | Top-n restaurants for a group |
| GET | `/business/{id}` | Business details and dietary tags |
| GET | `/health` | Service health and model status |

**Example — Group recommendation with halal filter:**

```json
POST /recommend/group
{
    "user_ids": ["user_abc", "user_def", "user_ghi"],
    "top_n": 5,
    "filters": {
        "halal": true,
        "min_stars": 4.0,
        "max_distance_km": 10,
        "latitude": 34.0522,
        "longitude": -118.2437
    }
}
```

**Response includes social influence weights** showing each member's contribution to the group decision:

```json
{
    "group_size": 3,
    "member_influences": {
        "user_abc": 0.4521,
        "user_def": 0.3102,
        "user_ghi": 0.2377
    },
    "recommendations": [...]
}
```

## Setup

### Prerequisites

- Google Cloud project with BigQuery and Cloud Run enabled
- Python 3.10+
- Yelp Academic Dataset loaded into BigQuery

### Local Development

```bash
python -m venv sigr_env
source sigr_env/bin/activate
pip install -r requirements.txt
python mock_train_sigr.py
```

### Training (Google Colab)

```python
!git clone https://github.com/gudipudiadarsh51/AI_planner.git
!cd AI_planner && python -m sigr_trainer.task
```

### Deployment

```bash
cd app
bash deploy.sh
```

## Tech Stack

- **Data**: BigQuery, Google Cloud Storage
- **Training**: TensorFlow 2.13, Vertex AI, Google Colab (T4 GPU)
- **Serving**: FastAPI, Cloud Run, Docker
- **Language**: Python 3.10

## References

- Yin, H., Wang, Q., Zheng, K., Li, Z., Yang, J., & Zhou, X. (2019). *Social Influence-based Group Representation Learning for Group Recommendation*. ICDE 2019.
- Yelp Academic Dataset: https://www.yelp.com/dataset

## License

MIT
