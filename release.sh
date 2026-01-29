#!/bin/bash
set -e

# Configuration
IMAGE_NAME="ghcr.io/akeslo/podcast-ad-remover"
TAG="mobile-redesign"

echo "🚀 Building Docker Image: $IMAGE_NAME:$TAG"

# Build and Push
docker buildx build \
  --platform linux/amd64 \
  --push \
  -t $IMAGE_NAME:$TAG \
  .

echo "🧹 Cleaning up GHCR..."
./imgclean.sh

echo "✅ Build, Push, and Cleanup Complete!"
