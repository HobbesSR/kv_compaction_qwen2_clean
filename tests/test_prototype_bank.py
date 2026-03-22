from __future__ import annotations

import json
from pathlib import Path

from kv_compaction_clean.config import load_config
from kv_compaction_clean.context_loader import load_context_sample
from kv_compaction_clean.feature_harvester import harvest_teacher_forced_features
from kv_compaction_clean.prototype_bank import build_state_from_observations, write_state


def test_build_state_from_harvested_features_has_probe_coverage() -> None:
    config = load_config("configs/qwen25_smoke/qwen2_5_3b.yaml")
    sample = load_context_sample(config)
    harvest = harvest_teacher_forced_features(sample, config)

    state = build_state_from_observations(config, harvest.observations)

    assert len(state.entries) <= config.sketch.max_prototypes
    assert len({entry.prototype_id for entry in state.entries}) == len(state.entries)
    assert any(entry.layer == 12 and entry.head == 3 for entry in state.entries)
    assert any(entry.layer == 28 and entry.head == 7 for entry in state.entries)


def test_write_state_serializes_json(tmp_path: Path) -> None:
    config = load_config("configs/qwen25_smoke/qwen2_5_3b.yaml")
    sample = load_context_sample(config)
    harvest = harvest_teacher_forced_features(sample, config)
    state = build_state_from_observations(config, harvest.observations)
    output_path = tmp_path / "prototype_bank.json"

    write_state(state, output_path)

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["sketch_kind"] == "prototype_bank"
    assert payload["tap_point"] == "post_query_pre_head_merge"
    assert payload["entries"]
