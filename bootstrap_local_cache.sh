#!/usr/bin/env bash
set -euo pipefail

# One-time bootstrap: copy the clean semantic base from GitHub Actions run 33.
# This downloads an existing artifact; it does NOT start a new Actions run.

REPO="HaSuNo0620/irui-analysis"
RUN_ID="34430083997"
ARTIFACT="irui-pilot-results"

mkdir -p results

echo "Downloading cached semantic base from run ${RUN_ID}..."
gh run download "${RUN_ID}" \
  --repo "${REPO}" \
  --name "${ARTIFACT}" \
  --dir results

for f in candidates.csv chunk_manifest.csv embeddings_raw.npy embeddings_anonymized.npy; do
  test -f "results/${f}" || {
    echo "Missing results/${f} after download" >&2
    exit 2
  }
done

echo "Local semantic cache is ready."
echo "Next: python run_analysis.py"
