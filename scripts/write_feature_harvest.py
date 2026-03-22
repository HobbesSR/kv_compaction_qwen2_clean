from __future__ import annotations

import argparse
from pathlib import Path

from kv_compaction_clean.config import load_config
from kv_compaction_clean.context_loader import load_context_sample
from kv_compaction_clean.feature_harvester import harvest_teacher_forced_features, write_feature_harvest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write a clean smoke teacher-forced feature harvest artifact.")
    parser.add_argument("--config", type=Path, required=True, help="Path to a clean-repo YAML config.")
    parser.add_argument("--output", type=Path, required=True, help="Where to write the feature harvest artifact.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    sample = load_context_sample(config)
    harvest = harvest_teacher_forced_features(sample, config)
    output_path = write_feature_harvest(harvest, args.output)
    print(f"feature harvest: {output_path}")
    print(
        f"sample={harvest.sample_id} "
        f"observations={harvest.observation_count} "
        f"layers={harvest.observed_layers} "
        f"heads={harvest.observed_heads}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
