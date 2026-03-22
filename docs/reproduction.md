# Reproduction

## Environment

The clean repo currently assumes:

- Python `3.9+`
- `torch`
- `transformers`
- `PyYAML`

Recommended install:

```bash
cd clean_repo
python3 -m pip install . --user
```

This installs the console entry points:

- `kv-clean-smoke`
- `kv-clean-demo`
- `kv-clean-export-examples`

Editable installs may depend on a newer `pip` than the one available in some
older environments. If `python3 -m pip install -e .` falls back to legacy
`setup.py develop` and fails, use the standard install command above.

## Smoke Result

Run:

```bash
cd clean_repo
kv-clean-smoke
```

Output artifact:

- `artifacts/qwen25_smoke/behavioral_eval_phase2_clean_k6.json`

Current observed local summary on the default `Qwen/Qwen2.5-3B` config:

- reference:
  - runtime: `33.071423s`
  - central details preserved: `4/6`
  - hallucination runs: `0`
- sketch:
  - runtime: `33.427049s`
  - central details preserved: `4/6`
  - hallucination runs: `0`
  - effective compact tokens: `54`
- control:
  - runtime: `31.943599s`
  - central details preserved: `4/6`
  - hallucination runs: `0`
  - effective compact tokens: `60`

These are the exact numbers from the current local demonstration run, not paper
claims and not guaranteed cross-machine constants.

## Service Demo

Run:

```bash
cd clean_repo
kv-clean-demo
```

The demo prints ingest progress during boundary collection, writes:

- `artifacts/qwen25_smoke/service_demo_summary.json`

and then accepts:

- `/compact <prompt>`
- `/full <prompt>`
- `/status`
- `/quit`

Current observed local summary:

- prefix tokens: `7168`
- preserved tail tokens: `1024`
- capture token count: `28`
- monitored observations: `336`
- monitored query samples: `336`
- compacted heads: `9`
- effective compact tokens: `54`

One demonstrated compacted answer runtime on the current lane:

- `10.053778s`

## Artifact Policy

`artifacts/` is intentionally ignored in git.

The clean repo policy is:

- check in code, configs, datasets, and docs
- regenerate artifacts locally from the documented commands
- keep a tiny checked-in summary set under `examples/qwen25_smoke/`
- document the current observed outputs in this file and the README

Intermediate local artifacts such as:

- `model_runtime_plan.json`
- `teacher_forced_feature_harvest.json`
- `prototype_bank.json`
- `query_coreset.json`

are useful for inspection and debugging, but they are not part of the curated
checked-in output surface.

If later we want golden artifacts in version control, that should be a small,
explicitly curated set rather than the entire local output directory.

## Checked-In Example Summaries

The repo includes a curated summary pair:

- `examples/qwen25_smoke/behavioral_eval_summary.json`
- `examples/qwen25_smoke/service_demo_summary.json`

These are derived from the local demonstration artifacts and are intended to be
small, stable reference outputs.

To refresh them after a new validated run:

```bash
cd clean_repo
kv-clean-export-examples
```
