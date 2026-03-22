# Clean Repo Plan

## Goal

Produce a small standalone artifact that demonstrates the compaction strategy on
validated smoke surfaces without carrying the full research history.

## Migration Order

1. `qwen25_smoke`
   - context loader
   - minimal runtime loader
   - teacher-forced boundary collection
   - prototype-bank / coreset path
   - highest-attention key selection
   - beta / Cv fitting
   - behavioral eval

2. `qwen35_smoke_v1`
   - same protocol
   - native prompt surface
   - native model configs
   - overlap prompts only for continuity checks

## Explicit Non-Goals

- no uncertainty branches
- no relational-binding branch in the first cut
- no service-loop prototype code
- no exploratory diagnostics unless promoted to stable docs

## Criteria For First Extraction

- Qwen2.5-3B smoke path reproduces the validated local result
- configs rerun from scratch in the clean subproject
- docs fit in a short reader-facing summary
