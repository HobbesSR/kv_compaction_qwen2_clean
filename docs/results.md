# Results

## Supported Claim

This repo currently supports one clean claim:

- on the local `qwen25_smoke` surface with `Qwen2.5-3B`, the compacted sketch
  matches the full-cache reference on the current demonstrated run with respect
  to:
  - central-detail preservation count
  - hallucination count

It also shows that the same compacted runtime can be used in an interactive
boundary-triggered demo path.

## Current Demonstrated Output

Smoke summary:

- prompt set: `phase2_clean`
- keys per head: `6`
- key selection: `highest_attention`
- reference:
  - runtime: `33.071423s`
  - central details preserved: `4/6`
  - hallucination runs: `0`
- sketch:
  - runtime: `33.427049s`
  - central details preserved: `4/6`
  - hallucination runs: `0`
  - effective compact tokens: `54`
  - compacted heads: `9`
- control:
  - runtime: `31.943599s`
  - central details preserved: `4/6`
  - hallucination runs: `0`
  - effective compact tokens: `60`
  - compacted heads: `10`

Service demo summary:

- prefix tokens: `7168`
- preserved tail tokens: `1024`
- capture token count: `28`
- monitored observations: `336`
- monitored query samples: `336`
- compacted heads: `9`
- effective compact tokens: `54`

## Where These Numbers Live

- checked-in summary:
  - `examples/qwen25_smoke/behavioral_eval_summary.json`
  - `examples/qwen25_smoke/service_demo_summary.json`
- regenerable full local artifacts:
  - `artifacts/qwen25_smoke/behavioral_eval_phase2_clean_k6.json`
  - `artifacts/qwen25_smoke/service_demo_summary.json`

## Non-Claims

This repo does not currently claim:

- broad benchmark superiority
- best-possible runtime optimization
- universality across model families
- coverage of the research repo's relational-binding failure analysis

Those belong to the research repo or later parallel stories.
