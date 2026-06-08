# manual_rag_v3

Same sources as `manual_rag_v2` (22 sources, no CUTLASS/CUDA), but with **whole-document chunks**.

## Key difference from v2

| | v2 | v3 |
|---|-----|-----|
| chunk_tokens | 1000 | 100000 |
| chunks per doc | 2-5 | **1** |
| matmul tutorial | truncated (store-only fragment) | **complete** (load + dot + store + K-loop) |

By not splitting documents into small chunks, each retrieved document is
self-contained. The model sees the **full kernel implementation** instead of
a partial snippet, eliminating the "improvise from fragment" failure mode.

## Usage

```bash
cd kernel/rag/manual_rag_v3
bash build_manual_rag_kb_v3.sh
```

Then in eval script:
```bash
RAG_KB_INDEX_PATH="kernel/rag/manual_rag_v3/processed/bm25_index.json"
```
