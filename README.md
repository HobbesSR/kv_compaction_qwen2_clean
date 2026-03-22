# KV Compaction Clean Repo

This is the presentable artifact lane for the KV compaction project. It lives
inside the research workspace for convenience today, but it is structured to be
split into its own repository cleanly.

## What This Repo Is For

This repo should let an outside user do two things without reading the research
log first:

1. Reproduce a small validated result.
2. Interact with a qualitatively service-viable compaction pipeline and watch
   long context being processed.

That means this repo is not a notebook graveyard and not a copy of every branch
from the research repo. Only stable, defended implementation surfaces move
here.

## Deliverables

### 1. Reproducible Smoke Result

The first stable result is:

- model family: `Qwen2.5`
- model size: `3B`
- task surface: `qwen25_smoke`
- paths:
  - full-cache reference
  - compacted sketch
  - explicit control baseline

The clean success criterion is: sketch reproduces the validated reference
behavior on the local smoke surface from a fresh checkout and config run.

### 2. Interactive Service Demo

The second deliverable is not a benchmark. It is a small interactive path that
shows:

- long context ingestion
- boundary-triggered compaction
- visible progress while evidence is collected
- continued answering over compacted state

This path should feel service-shaped even if it is still local and offline.

## Repo Principles

- no runtime dependency on the parent research package
- only validated surfaces move here
- one default path first, variants second
- paper baselines are explicit controls, not hidden inside the main method
- Qwen2.5 and Qwen3.5 are parallel stories sharing infrastructure, not one
  prompt surface forced onto both

## Planned Tracks

- `qwen25_smoke`
  - first writeup-quality story
  - validated local result
  - Qwen2.5-3B
- `qwen35_smoke_v1`
  - native evaluation surface
  - same compaction protocol
  - overlap prompts used only for continuity checks
- `service_demo`
  - interactive local path
  - visible context ingestion and compaction boundary
  - minimal UI/CLI before any heavier front-end work

## Layout

- `src/kv_compaction_clean/`
  - extracted implementation surface
- `configs/`
  - stable runnable configs only
- `data/`
  - clean smoke datasets only
- `docs/`
  - short outside-reader documentation
- `tests/`
  - focused validation for the clean path

See [docs/plan.md](./docs/plan.md) for migration order and
[docs/repo_contract.md](./docs/repo_contract.md) for scope boundaries.
