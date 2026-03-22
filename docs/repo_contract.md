# Repo Contract

## What Belongs Here

- validated implementation needed to rerun the clean smoke result
- stable configs and datasets used by that result
- a small interactive path that demonstrates service-shaped compaction
- concise docs that explain how to run the repo and what claims it supports

## What Does Not Belong Here

- exploratory branches that do not change the validated story
- one-off diagnostics
- large experimental ablation ladders
- implementation paths that are only useful for research archaeology
- any runtime import from the parent research repo

## Evaluation Contract

Every evaluation story in this repo should state:

- the model
- the prompt surface
- the controls
- the exact claimed result

The clean repo should not rely on the reader to infer which branch was the real
one.

## Service Demo Contract

The interactive path is allowed to be minimal, but it must demonstrate:

- extended context ingestion
- visible progress while ingesting
- a compaction boundary
- answering after compaction

It should not silently fall back to a trivial non-compacting path.

## Parallel Model Families

Different model families may share infrastructure here, but each family should
have its own native evaluation surface if prompt alignment differs materially.

The clean repo should prefer:

- shared protocol
- separate calibration

over pretending one model family's prompt ecology is universal.
