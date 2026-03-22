from __future__ import annotations

import argparse
from pathlib import Path

from kv_compaction_clean.config import load_config
from kv_compaction_clean.context_loader import load_context_sample
from kv_compaction_clean.coreset import extract_query_coreset, write_query_coreset
from kv_compaction_clean.feature_harvester import harvest_teacher_forced_features
from kv_compaction_clean.prototype_bank import build_state_from_observations, write_state


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write clean smoke prototype-bank and query-coreset artifacts.")
    parser.add_argument("--config", type=Path, required=True, help="Path to a clean-repo YAML config.")
    parser.add_argument("--state-output", type=Path, required=True, help="Where to write the prototype-bank artifact.")
    parser.add_argument("--coreset-output", type=Path, required=True, help="Where to write the query-coreset artifact.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    sample = load_context_sample(config)
    harvest = harvest_teacher_forced_features(sample, config)
    state = build_state_from_observations(config, harvest.observations)
    coreset = extract_query_coreset(sample.sample_id, sample.boundary.boundary_id, state, config)
    state_output = write_state(state, args.state_output)
    coreset_output = write_query_coreset(coreset, args.coreset_output)
    print(f"prototype bank: {state_output}")
    print(f"query coreset: {coreset_output}")
    print(f"entries={len(state.entries)} coreset_entries={len(coreset.selected_entries)} total_weight={coreset.total_weight}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
