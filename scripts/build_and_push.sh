#!/bin/bash
# Build and push the training/inference Docker image.
# Usage: ./scripts/build_and_push.sh <registry>
#   e.g.: ./scripts/build_and_push.sh gcr.io/my-project

set -euo pipefail

REGISTRY="${1:?Usage: $0 <registry>}"
IMAGE_NAME="ai-audio-detection"
TAG="latest"

echo "Building ${IMAGE_NAME}:${TAG}..."
docker build -t "${IMAGE_NAME}:${TAG}" .

echo "Tagging for registry: ${REGISTRY}/${IMAGE_NAME}:${TAG}"
docker tag "${IMAGE_NAME}:${TAG}" "${REGISTRY}/${IMAGE_NAME}:${TAG}"

echo "Pushing to ${REGISTRY}..."
docker push "${REGISTRY}/${IMAGE_NAME}:${TAG}"

echo "Done. Image available at: ${REGISTRY}/${IMAGE_NAME}:${TAG}"
echo "Update k8s/training-job.yaml and k8s/inference-deployment.yaml with this image path."
