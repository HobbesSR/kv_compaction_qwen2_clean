# Reproduction

## Environment

The clean repo currently assumes:

- Python `3.9+`
- `torch`
- `transformers`
- `PyYAML`

Install editable local deps from the nested repo root:

```bash
cd clean_repo
pip install -e .
```

## Smoke Result

Run:

```bash
cd clean_repo
PYTHONPATH=src python scripts/run_behavioral_eval.py
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
PYTHONPATH=src python scripts/run_service_demo.py
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
- document the current observed outputs in this file and the README

If later we want golden artifacts in version control, that should be a small,
explicitly curated set rather than the entire local output directory.
