#!/usr/bin/env bash
# docker/rebuild.sh — One-command Docker rebuild with health verification
# Usage: ./docker/rebuild.sh [--no-cache] [--skip-verify]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

NO_CACHE=""
SKIP_VERIFY=false

for arg in "$@"; do
  case "$arg" in
    --no-cache) NO_CACHE="--no-cache" ;;
    --skip-verify) SKIP_VERIFY=true ;;
    *) echo "Unknown arg: $arg"; exit 1 ;;
  esac
done

echo "=========================================="
echo "  Product Fatigue — Docker Rebuild"
echo "=========================================="

# 1. Pre-flight checks
echo ""
echo "[1/5] Pre-flight checks..."
if ! command -v docker &>/dev/null; then
  echo "ERROR: docker not found. Install Docker first."
  exit 1
fi
if ! docker info &>/dev/null 2>&1; then
  echo "ERROR: Docker daemon is not running."
  exit 1
fi

# Check required model files exist
MISSING=0
for modality in reviews sales usage; do
  if [ ! -f "models/${modality}_artifacts.pkl" ]; then
    echo "WARNING: models/${modality}_artifacts.pkl not found — run training first"
    MISSING=1
  fi
done
if [ "$MISSING" -eq 1 ]; then
  echo "Some model artifacts are missing. The API will start in degraded mode."
fi

# 2. Stop existing containers
echo ""
echo "[2/5] Stopping existing containers..."
docker compose -f docker/docker-compose.yml down --remove-orphans 2>/dev/null || true

# 3. Build images
echo ""
echo "[3/5] Building Docker images... $NO_CACHE"
docker compose -f docker/docker-compose.yml build $NO_CACHE

# 4. Start services
echo ""
echo "[4/5] Starting services..."
docker compose -f docker/docker-compose.yml up -d

# 5. Post-build verification
if [ "$SKIP_VERIFY" = true ]; then
  echo ""
  echo "[5/5] Skipping verification (--skip-verify)"
else
  echo ""
  echo "[5/5] Running post-build verification..."
  bash "$SCRIPT_DIR/verify.sh"
fi

echo ""
echo "=========================================="
echo "  Rebuild complete!"
echo "=========================================="
echo ""
echo "Services:"
echo "  API:        http://localhost:8000"
echo "  API Docs:   http://localhost:8000/docs"
echo "  MLflow:     http://localhost:5001"
echo "  Prometheus: http://localhost:9090"
echo "  Grafana:    http://localhost:3000 (admin/admin)"
echo ""
