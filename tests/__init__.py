"""Test package for Memory Mesh.

Modules:
- ``test_focused_fixes`` — Batch 1 (HTTP security, body limits, Memory embedder/timestamps)
- ``test_batch2_fixes`` — Batch 2 (OpenCode, sync, summaries, FKs, shell safety, …)
- ``test_batch3_fixes`` — Batch 3 (paths, embeddings, provider scan, failure visibility, …)
- ``test_batch4_fixes`` — Batch 4 (T06 facade unification, T18 bundle summaries)
- ``test_batch4_review`` — Batch 4 final review (concurrency, legacy RO, schema, reports)
- ``test_opencode_plugin_runtime`` — compiled OpenCode plugin (requires Node + pinned esbuild)
- ``helpers`` — shared temporary HOME / legacy DB helpers
"""
