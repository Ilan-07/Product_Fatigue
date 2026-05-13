#!/usr/bin/env bash
# docker/verify.sh — Post-rebuild health verification
# Checks that all Docker services are running and responding correctly.
set -euo pipefail

API_URL="http://localhost:8000"
MAX_RETRIES=30
RETRY_INTERVAL=2

echo "Running post-rebuild verification..."
echo ""

# Helper: wait for an endpoint to return HTTP 200
wait_for_endpoint() {
  local url="$1"
  local name="$2"
  local retries=0

  while [ $retries -lt $MAX_RETRIES ]; do
    if curl -sf "$url" > /dev/null 2>&1; then
      echo "  [PASS] $name is up ($url)"
      return 0
    fi
    retries=$((retries + 1))
    sleep $RETRY_INTERVAL
  done
  echo "  [FAIL] $name did not respond after $((MAX_RETRIES * RETRY_INTERVAL))s ($url)"
  return 1
}

FAILURES=0

# 1. API health check
echo "--- Service Health ---"
wait_for_endpoint "$API_URL/health" "API Health" || FAILURES=$((FAILURES + 1))

# 2. Pipeline status
echo ""
echo "--- Pipeline Status ---"
PIPELINE_STATUS=$(curl -sf "$API_URL/pipeline/status" 2>/dev/null || echo '{"error": true}')
if echo "$PIPELINE_STATUS" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('pipeline_status','error'))" 2>/dev/null | grep -q "operational"; then
  echo "  [PASS] Pipeline status: operational"
else
  echo "  [WARN] Pipeline status: degraded or unavailable"
  echo "         $PIPELINE_STATUS" | head -c 200
fi

# 3. Model info
echo ""
echo "--- Model Info ---"
MODEL_INFO=$(curl -sf "$API_URL/model/info" 2>/dev/null || echo '{"error": true}')
for modality in reviews sales usage; do
  if echo "$MODEL_INFO" | python3 -c "import sys,json; d=json.load(sys.stdin); m=d.get('models',{}).get('$modality',{}); print(m.get('champion_model','missing'))" 2>/dev/null | grep -qv "missing"; then
    echo "  [PASS] $modality model loaded"
  else
    echo "  [WARN] $modality model not loaded"
  fi
done

# 4. Prediction smoke test (reviews)
echo ""
echo "--- Smoke Test: Reviews Prediction ---"
SMOKE_RESULT=$(curl -sf -X POST "$API_URL/predict/reviews" \
  -H "Content-Type: application/json" \
  -d '{
    "sentiment_mean": 0.62, "sentiment_std": 0.21, "review_count": 24,
    "score_min": 2, "score_max": 5, "score_median": 3.5,
    "product_age_months": 14, "sentiment_polarization": 1.2,
    "reviewer_diversity_change": -8
  }' 2>/dev/null || echo '{"error": true}')

if echo "$SMOKE_RESULT" | python3 -c "import sys,json; d=json.load(sys.stdin); assert 'fatigue_status' in d; print(d['fatigue_status'])" 2>/dev/null; then
  echo "  [PASS] Reviews prediction returned successfully"
else
  echo "  [FAIL] Reviews prediction failed"
  FAILURES=$((FAILURES + 1))
fi

# 5. V1/V2 versioned routes
echo ""
echo "--- Versioned Routes ---"
V1_HEALTH=$(curl -sf "$API_URL/v1/health" 2>/dev/null || echo '{}')
V2_HEALTH=$(curl -sf "$API_URL/v2/health" 2>/dev/null || echo '{}')

if echo "$V1_HEALTH" | python3 -c "import sys,json; d=json.load(sys.stdin); assert d.get('api_version')=='v1'" 2>/dev/null; then
  echo "  [PASS] V1 routes accessible"
else
  echo "  [WARN] V1 routes not accessible"
fi

if echo "$V2_HEALTH" | python3 -c "import sys,json; d=json.load(sys.stdin); assert d.get('api_version')=='v2'" 2>/dev/null; then
  echo "  [PASS] V2 routes accessible"
else
  echo "  [WARN] V2 routes not accessible"
fi

# 6. Prometheus metrics endpoint
echo ""
echo "--- Observability ---"
wait_for_endpoint "$API_URL/metrics" "Prometheus Metrics" || FAILURES=$((FAILURES + 1))

# Summary
echo ""
echo "=========================================="
if [ $FAILURES -eq 0 ]; then
  echo "  All checks passed!"
else
  echo "  $FAILURES check(s) failed. Review output above."
fi
echo "=========================================="
exit $FAILURES
