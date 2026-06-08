#!/bin/bash
set -euo pipefail

KB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
V1_DIR="${KB_DIR}/../manual_rag"
RAG_DIR="${KB_DIR}/.."

# Copy raw files from v1
if [ ! -d "${KB_DIR}/raw" ] || [ -z "$(ls -A "${KB_DIR}/raw" 2>/dev/null)" ]; then
  mkdir -p "${KB_DIR}/raw"
  cp "${V1_DIR}/raw/"*.html "${KB_DIR}/raw/" 2>/dev/null || true
fi

# Filter v1's downloaded_manifest to v3 source IDs
python3 -c "
import json, os

with open('${V1_DIR}/downloaded_manifest.json') as f:
    v1 = json.load(f)

with open('${KB_DIR}/manual_manifest_v3.json') as f:
    v3_manifest = json.load(f)

v3_ids = {s['id'] for s in v3_manifest['sources']}
filtered = [s for s in v1['sources'] if s['id'] in v3_ids]

for s in filtered:
    name = os.path.basename(s['local_path'])
    s['local_path'] = f'${KB_DIR}/raw/' + name

with open('${KB_DIR}/downloaded_manifest.json', 'w') as f:
    json.dump({'sources': filtered}, f, ensure_ascii=False, indent=2)

print(f'Filtered manifest: {len(filtered)} sources')
"

# Build with NO chunking: each document is one huge chunk
python3 "${RAG_DIR}/build_manual_kb.py" \
  --downloaded-manifest "${KB_DIR}/downloaded_manifest.json" \
  --output-dir "${KB_DIR}/processed" \
  --chunk-tokens 100000 \
  --overlap-tokens 0

echo ""
echo "v3 KB built at: ${KB_DIR}/processed/bm25_index.json"
echo "Strategy: whole-document chunks (chunk_tokens=100000, overlap=0)"
echo "Sources: 22 (no CUTLASS/CUDA)"
