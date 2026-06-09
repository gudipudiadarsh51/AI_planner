#!/bin/bash
# deploy.sh — Deploy SIGR Recommendation API to Cloud Run
# Run this from Cloud Shell after training completes

set -e

PROJECT_ID="future-area-496000-v2"
REGION="us-central1"
SERVICE_NAME="sigr-recommender"
IMAGE_NAME="gcr.io/${PROJECT_ID}/${SERVICE_NAME}"

echo "=== Step 1: Build container image ==="
cd ~/AI_planner/app
gcloud builds submit --tag ${IMAGE_NAME} .

echo "=== Step 2: Deploy to Cloud Run ==="
gcloud run deploy ${SERVICE_NAME} \
  --image ${IMAGE_NAME} \
  --region ${REGION} \
  --platform managed \
  --allow-unauthenticated \
  --memory 4Gi \
  --cpu 2 \
  --min-instances 0 \
  --max-instances 3 \
  --timeout 300 \
  --set-env-vars "GCS_BUCKET=yelp-sigr-training,MODEL_PREFIX=models/sigr_v1,BIZ_PREFIX=sigr-training/business_features"

echo "=== Deployment complete ==="
echo ""
echo "API URL:"
gcloud run services describe ${SERVICE_NAME} \
  --region ${REGION} \
  --format="value(status.url)"
