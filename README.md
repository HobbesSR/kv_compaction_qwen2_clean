# KV Compaction Clean Repo

This directory is a standalone clean-artifact scaffold that lives inside the
research repo for convenience. It is intended to become its own repository once
the validated surface is stable.

Principles:

- no runtime dependency on the parent research package
- only validated surfaces move here
- Qwen2.5 and Qwen3.5 are parallel stories sharing infrastructure, not one
  pipeline patched to imitate the other

Planned tracks:

- `qwen25_smoke`
  - validated local story
  - Qwen2.5-3B
  - reference vs sketch vs control
- `qwen35_smoke_v1`
  - native calibration surface
  - paper-aligned controls on a Qwen3.5-local task family
  - overlap slice used only for pipeline continuity checks

Top-level structure:

- `src/kv_compaction_clean/`
  - implementation surface that will be copied in deliberately
- `configs/`
  - stable runnable configs only
- `data/`
  - clean smoke datasets only
- `docs/`
  - concise outside-reader documentation
- `tests/`
  - focused validation for the clean path

This subproject is intentionally empty of copied implementation for now. The
next step is to migrate the minimal validated Qwen2.5 path into it.
