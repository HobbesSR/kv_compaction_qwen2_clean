# Clean Repo Plan

## Goal

Produce a small standalone artifact that demonstrates the compaction strategy on
validated smoke surfaces and exposes a service-shaped interactive path, without
carrying the full research history.

## First Public Story

The first public story is `qwen25_smoke`:

- `Qwen2.5-3B`
- validated smoke surface
- full-cache reference vs compacted sketch vs explicit control
- one-command rerun from clean configs

This is the first thing that must work end to end.

## Second Story

The second story is `service_demo`:

- load an extended context
- show ingestion progress
- compact at a boundary
- continue interaction over compacted state

This should demonstrate service viability qualitatively, even before any richer
front-end layer exists.

## Parallel Story

`qwen35_smoke_v1` is allowed in the clean repo once it is natively calibrated.
It is not an appendix to `qwen25_smoke`; it is a parallel evaluation story with
shared infrastructure.

## Migration Order

1. `qwen25_smoke` extraction
   - config loader
   - context loader
   - minimal runtime loader
   - teacher-forced boundary collection
   - prototype-bank / coreset path
   - highest-attention key selection
   - beta / Cv fitting
   - behavioral eval
2. `service_demo` shell
   - minimal interaction loop
   - visible ingestion progress
   - boundary compaction event
   - continued answering
3. `qwen35_smoke_v1`
   - native prompt surface
   - native model configs
   - overlap prompts only for continuity checks

## Explicit Non-Goals

- no uncertainty branches
- no relational-binding branch in the first cut
- no exploratory diagnostics unless promoted to stable docs
- no hard dependency on the parent research repo
- no paper-submodule coupling in the main path

## Criteria For First Extraction

- Qwen2.5-3B smoke path reproduces the validated local result
- configs rerun from scratch in this subproject
- an outside user can tell what to run from the README alone
- the implementation surface is small enough to explain in a short architecture
  note

## Current State

`qwen25_smoke` now exists as a real runnable path:

- `scripts/run_behavioral_eval.py`
- `scripts/run_service_demo.py`

What remains is polish rather than missing architecture:

- tighten the architecture note around the end-to-end path
- decide which local artifacts should be checked in as golden examples
- keep `qwen35_smoke_v1` separate until it has a native calibrated surface
