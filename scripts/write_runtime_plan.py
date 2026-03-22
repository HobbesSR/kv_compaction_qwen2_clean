from __future__ import annotations

import argparse
from pathlib import Path

from kv_compaction_clean.config import load_config
from kv_compaction_clean.context_loader import load_context_sample
from kv_compaction_clean.model_runtime import build_model_runtime_plan, write_model_runtime_plan


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write a clean smoke runtime plan artifact.")
    parser.add_argument("--config", type=Path, required=True, help="Path to a clean-repo YAML config.")
    parser.add_argument("--output", type=Path, required=True, help="Where to write the runtime plan artifact.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    sample = load_context_sample(config)
    plan = build_model_runtime_plan(sample, config)
    output_path = write_model_runtime_plan(plan, args.output)
    print(f"runtime plan: {output_path}")
    print(
        f"model={plan.huggingface_id} "
        f"context_tokens={plan.logical_context_tokens} "
        f"expected_prefill_chunks={plan.expected_prefill_chunks}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
