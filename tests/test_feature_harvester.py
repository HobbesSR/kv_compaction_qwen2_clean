from __future__ import annotations

import json
from pathlib import Path

from kv_compaction_clean.config import load_config
from kv_compaction_clean.context_loader import load_context_sample
from kv_compaction_clean.feature_harvester import (
    harvest_teacher_forced_features,
    write_feature_harvest,
)


def test_harvest_teacher_forced_features_matches_frozen_schema() -> None:
    config = load_config("configs/qwen25_smoke/qwen2_5_3b.yaml")
    sample = load_context_sample(config)

    harvest = harvest_teacher_forced_features(sample, config)

    assert harvest.sample_id == sample.sample_id
    assert harvest.logical_context_tokens == 8192
    assert harvest.physical_context_tokens == 8192
    assert harvest.observed_layers == [4, 12, 20, 28]
    assert harvest.observed_heads == [0, 3, 7]
    assert harvest.observation_count == len(harvest.observations)
    assert harvest.observation_count == 168
    assert all(
        len(observation.query_projection) == config.feature_schema.query_projection_dim
        for observation in harvest.observations
    )
    assert all(
        len(observation.output_projection) == config.feature_schema.output_projection_dim
        for observation in harvest.observations
    )
    assert any(observation.source_turn_id == "turn_7" for observation in harvest.observations)


def test_write_feature_harvest_serializes_json(tmp_path: Path) -> None:
    config = load_config("configs/qwen25_smoke/qwen2_5_3b.yaml")
    sample = load_context_sample(config)
    harvest = harvest_teacher_forced_features(sample, config)
    output_path = tmp_path / "teacher_forced_feature_harvest.json"

    write_feature_harvest(harvest, output_path)

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["tap_point"] == "post_query_pre_head_merge"
    assert payload["observation_count"] == 168
    assert payload["observations"][0]["source_speaker"]
