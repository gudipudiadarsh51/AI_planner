"""
Test the SIGR Recommendation API after deployment.

Usage:
    python test_api.py https://sigr-recommender-xxxxx.run.app

Requires the API URL from Cloud Run deployment.
"""

import sys
import json
import requests

if len(sys.argv) < 2:
    print("Usage: python test_api.py <API_URL>")
    print("Example: python test_api.py https://sigr-recommender-abc123.run.app")
    sys.exit(1)

BASE_URL = sys.argv[1].rstrip("/")
print(f"Testing API at: {BASE_URL}\n")

# ── Test 1: Health check ──────────────────────────────────────────────────
print("=== Test 1: Health Check ===")
r = requests.get(f"{BASE_URL}/health")
print(f"Status: {r.status_code}")
print(json.dumps(r.json(), indent=2))
print()

# ── Test 2: Single user recommendation ────────────────────────────────────
print("=== Test 2: User Recommendation ===")
# Use a sample user_id from the dataset
r = requests.post(f"{BASE_URL}/recommend/user", json={
    "user_id": "hG7b0MtEbXx5QzbzE6C_VA",  # Replace with a real user_id
    "top_n": 5
})
print(f"Status: {r.status_code}")
if r.status_code == 200:
    data = r.json()
    print(f"Recommendations for user:")
    for i, rec in enumerate(data["recommendations"]):
        print(f"  {i+1}. {rec['name']} — score: {rec['score']}, "
              f"stars: {rec.get('stars', 'N/A')}, "
              f"city: {rec.get('city', 'N/A')}")
elif r.status_code == 404:
    print("User not found — try a different user_id from your dataset")
    print(r.json())
print()

# ── Test 3: User recommendation with filters ──────────────────────────────
print("=== Test 3: Filtered User Recommendation ===")
r = requests.post(f"{BASE_URL}/recommend/user", json={
    "user_id": "hG7b0MtEbXx5QzbzE6C_VA",
    "top_n": 5,
    "filters": {
        "min_stars": 4.0,
        "city": "Los Angeles"
    }
})
print(f"Status: {r.status_code}")
if r.status_code == 200:
    data = r.json()
    print(f"Filtered recommendations:")
    for i, rec in enumerate(data["recommendations"]):
        print(f"  {i+1}. {rec['name']} — score: {rec['score']}, "
              f"stars: {rec.get('stars', 'N/A')}, "
              f"halal: {rec.get('tag_halal', 'N/A')}")
print()

# ── Test 4: Group recommendation ──────────────────────────────────────────
print("=== Test 4: Group Recommendation ===")
r = requests.post(f"{BASE_URL}/recommend/group", json={
    "user_ids": [
        "hG7b0MtEbXx5QzbzE6C_VA",
        "nVKOMMkvgyhTPOIG_etKbA",
        "CxDOIDnH8gp9KXzpBHJYXw"
    ],
    "top_n": 5
})
print(f"Status: {r.status_code}")
if r.status_code == 200:
    data = r.json()
    print(f"Group size: {data['group_size']}")
    print(f"Member influences: {json.dumps(data['member_influences'], indent=2)}")
    print(f"Recommendations:")
    for i, rec in enumerate(data["recommendations"]):
        print(f"  {i+1}. {rec['name']} — score: {rec['score']}, "
              f"stars: {rec.get('stars', 'N/A')}")
print()

# ── Test 5: Group recommendation with halal filter ────────────────────────
print("=== Test 5: Group + Halal Filter ===")
r = requests.post(f"{BASE_URL}/recommend/group", json={
    "user_ids": [
        "hG7b0MtEbXx5QzbzE6C_VA",
        "nVKOMMkvgyhTPOIG_etKbA"
    ],
    "top_n": 5,
    "filters": {
        "halal": True,
        "min_stars": 3.5
    }
})
print(f"Status: {r.status_code}")
if r.status_code == 200:
    data = r.json()
    print(f"Halal-filtered group recommendations:")
    for i, rec in enumerate(data["recommendations"]):
        print(f"  {i+1}. {rec['name']} — halal: {rec.get('tag_halal')}, "
              f"score: {rec['score']}")
print()

# ── Test 6: Location-based filtering ──────────────────────────────────────
print("=== Test 6: Location-based Filtering ===")
r = requests.post(f"{BASE_URL}/recommend/user", json={
    "user_id": "hG7b0MtEbXx5QzbzE6C_VA",
    "top_n": 5,
    "filters": {
        "latitude": 34.0522,
        "longitude": -118.2437,
        "max_distance_km": 5
    }
})
print(f"Status: {r.status_code}")
if r.status_code == 200:
    data = r.json()
    print(f"Nearby recommendations (within 5km):")
    for i, rec in enumerate(data["recommendations"]):
        print(f"  {i+1}. {rec['name']} — distance: {rec.get('distance_km', 'N/A')} km, "
              f"score: {rec['score']}")

print("\n=== All tests complete ===")
