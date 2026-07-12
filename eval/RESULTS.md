# Eval results

One row per eval run. Retrieval metrics exclude out_of_scope items.

| tag | date | recall@5 | mrr | faithfulness | correctness |
|-----|------|----------|-----|--------------|-------------|
| v1-naive-dirty-corpus | 2026-07-05 | 0.500 | 0.388 | 0.893 | 0.688 |
| v2-clean-corpus | 2026-07-05 | 0.538 | 0.424 | 0.897 | 0.667 |
| v2-clean-corpus-labelfix | 2026-07-05 | 0.577 | 0.463 | — | — |
| verify-by-hand | 2026-07-05 | 0.577 | 0.463 | — | — |
| v3-chunk-align-256 | 2026-07-05 | 0.654 | 0.428 | — | — |
| v3-chunk-align-bge | 2026-07-05 | 0.769 | 0.571 | — | — |
| v3-structure-chunks | 2026-07-05 | 0.615 | 0.466 | — | — |
| v3-structure-chunks-packed | 2026-07-05 | 0.615 | 0.435 | — | — |
| v3-hybrid-rrf | 2026-07-05 | 0.769 | 0.615 | — | — |
| v3-rerank | 2026-07-05 | 0.731 | 0.488 | — | — |
| v3-hnsw | 2026-07-05 | 0.769 | 0.615 | — | — |
| v3-final | 2026-07-05 | 0.769 | 0.615 | 0.903 | 0.673 |
| full-verify | 2026-07-05 | 0.769 | 0.615 | — | — |
| v4-query-rewrite | 2026-07-06 | 0.769 | 0.663 | — | — |
| v4-rewrite-rerank | 2026-07-06 | 0.731 | 0.506 | — | — |
| v4-state-filter | 2026-07-06 | 0.769 | 0.663 | — | — |
| v4-citations | 2026-07-06 | 0.769 | 0.578 | 0.828 | 0.626 |
| v4-final | 2026-07-06 | 0.769 | 0.578 | 0.860 | 0.727 |
| v4-verify-1 | 2026-07-07 | 0.769 | 0.578 | — | — |
| v4-verify-2 | 2026-07-07 | 0.769 | 0.578 | — | — |
| v5-final | 2026-07-08 | 0.769 | 0.578 | — | — |
| v6-agent-retrieval | 2026-07-10 | 0.769 | 0.578 | — | — |
| v6-pipeline-multihop | 2026-07-10 | 0.833 | 0.492 | 0.700 | 0.633 |
| v6-agent-multihop | 2026-07-10 | 0.833 | 0.492 | 0.817 | 0.700 |
| v7-pipeline-revised | 2026-07-11 | 0.776 | 0.594 | — | — |
| v7-agent-revised | 2026-07-11 | 0.741 | 0.570 | — | — |
