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

More precisely, this clean artifact currently demonstrates:

- teacher-forced sparse boundary evidence collection
- prototype-bank sketching over that boundary evidence
- comparison against an explicit teacher-forced control query source

It does not yet package the research repo's separate generation-time streaming
observer result.

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
- that the current clean path is the low-overhead streaming observer path from
  the research repo

## Streaming Observer Gap

The research repo now contains a stronger systems result: narrow live
generation-time observation can be run with very low overhead on the tested
surface. That result is intentionally not folded into this clean repo yet.

This repo's current public story is narrower:

- boundary compaction
- teacher-forced sparse boundary evidence
- sketch vs explicit teacher-forced control

That is a legitimate and reproducible baseline, but it is not the final
service-overhead claim from the research lane.

Those belong to the research repo or later parallel stories.
