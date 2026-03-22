from __future__ import annotations

import json
from pathlib import Path

from kv_compaction_clean.config import load_config
from kv_compaction_clean.context_loader import load_context_sample
from kv_compaction_clean.coreset import extract_query_coreset, write_query_coreset
from kv_compaction_clean.feature_harvester import harvest_teacher_forced_features
from kv_compaction_clean.prototype_bank import build_state_from_observations


def test_extract_query_coreset_selects_ranked_entries() -> None:
    config = load_config("configs/qwen25_smoke/qwen2_5_3b.yaml")
    sample = load_context_sample(config)
    harvest = harvest_teacher_forced_features(sample, config)
    state = build_state_from_observations(config, harvest.observations)

    coreset = extract_query_coreset(sample.sample_id, sample.boundary.boundary_id, state, config)

    assert coreset.source == "prototype_bank"
    assert coreset.max_entries <= config.sketch.max_prototypes
    assert len(coreset.selected_entries) == coreset.max_entries
    assert coreset.total_weight > 0.0


def test_write_query_coreset_serializes_json(tmp_path: Path) -> None:
    config = load_config("configs/qwen25_smoke/qwen2_5_3b.yaml")
    sample = load_context_sample(config)
    harvest = harvest_teacher_forced_features(sample, config)
    state = build_state_from_observations(config, harvest.observations)
    coreset = extract_query_coreset(sample.sample_id, sample.boundary.boundary_id, state, config)
    output_path = tmp_path / "query_coreset.json"

    write_query_coreset(coreset, output_path)

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["source"] == "prototype_bank"
    assert payload["selected_entries"]
