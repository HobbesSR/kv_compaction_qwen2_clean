from __future__ import annotations

"""Synthetic feature harvest substrate used only for early/local smoke tests.

This module does not read model attention or hidden states. It generates
deterministic hashed vectors from sample metadata and a fixed token stride so
unit tests and early substrate checks can run without a live model pass.

The real model-backed collection path in the clean repo is
`boundary_collection.py`. Outside readers should treat that module as the
authoritative evidence-collection implementation for the demonstrated smoke and
service paths.
"""

import hashlib
import json
import math
from pathlib import Path

from kv_compaction_clean.data_types import (
    ContextTurn,
    FeatureHarvest,
    FeatureObservation,
    LoadedContextSample,
    SmokeTestConfig,
)
from kv_compaction_clean.model_runtime import PROBE_HEADS, PROBE_LAYERS


TOKEN_STRIDE = 512


def _hash_to_unit_interval(seed: str) -> float:
    digest = hashlib.sha256(seed.encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], byteorder="big")
    return value / float((1 << 64) - 1)


def _hashed_vector(seed: str, dim: int) -> list[float]:
    raw = [(2.0 * _hash_to_unit_interval(f"{seed}:{index}")) - 1.0 for index in range(dim)]
    norm = math.sqrt(sum(value * value for value in raw)) or 1.0
    return [round(value / norm, 6) for value in raw]


def _turn_spans(sample: LoadedContextSample) -> list[tuple[int, int, ContextTurn]]:
    spans: list[tuple[int, int, ContextTurn]] = []
    cursor = 0
    for turn in sample.turns:
        next_cursor = cursor + turn.token_count
        spans.append((cursor, next_cursor, turn))
        cursor = next_cursor
    return spans


def _turn_for_token(sample: LoadedContextSample, token_index: int) -> ContextTurn:
    for start_token, end_token, turn in _turn_spans(sample):
        if start_token <= token_index < end_token:
            return turn
    return sample.turns[-1]


def _probe_token_indices(prefix_token_count: int) -> list[int]:
    indices = list(range(TOKEN_STRIDE - 1, prefix_token_count, TOKEN_STRIDE))
    last_prefix_token = prefix_token_count - 1
    if not indices or indices[-1] != last_prefix_token:
        indices.append(last_prefix_token)
    return indices


def _speaker_mass_bias(speaker: str) -> float:
    if speaker == "assistant":
        return 0.11
    if speaker == "user":
        return 0.08
    if speaker == "tool":
        return 0.09
    return 0.06


def _keyword_mass_bonus(content: str) -> float:
    lowered = content.lower()
    keywords = (
        "relay",
        "rollback",
        "controller",
        "inventory",
        "checklist",
        "supplier",
    )
    matches = sum(1 for keyword in keywords if keyword in lowered)
    return min(0.06, matches * 0.01)


def _prefix_mass_share(
    sample: LoadedContextSample,
    token_index: int,
    layer: int,
    head: int,
    turn_content: str,
    speaker: str,
) -> float:
    progress = (token_index + 1) / sample.boundary.prefix_token_count
    share = (
        _speaker_mass_bias(speaker)
        + (0.12 * progress)
        + _keyword_mass_bonus(turn_content)
        + (0.01 * ((layer % 8) / 7.0))
        + (0.01 * (head / max(PROBE_HEADS)))
    )
    return round(min(0.35, max(0.04, share)), 6)


def harvest_teacher_forced_features(sample: LoadedContextSample, config: SmokeTestConfig) -> FeatureHarvest:
    observations: list[FeatureObservation] = []
    token_indices = _probe_token_indices(sample.boundary.prefix_token_count)

    for token_index in token_indices:
        turn = _turn_for_token(sample, token_index)
        for layer in PROBE_LAYERS:
            for head in PROBE_HEADS:
                basis = (
                    f"{sample.sample_id}|{turn.turn_id}|{token_index}|"
                    f"{layer}|{head}|{turn.speaker}|{config.experiment.seed}"
                )
                prefix_mass_share = _prefix_mass_share(
                    sample,
                    token_index,
                    layer,
                    head,
                    turn.content,
                    turn.speaker,
                )
                observations.append(
                    FeatureObservation(
                        token_index=token_index,
                        layer=layer,
                        head=head,
                        tap_point=config.feature_schema.tap_point,
                        query_projection=_hashed_vector(
                            f"{basis}|query|{turn.content}",
                            config.feature_schema.query_projection_dim,
                        ),
                        prefix_mass_share=prefix_mass_share,
                        raw_prefix_mass=round((token_index + 1) * prefix_mass_share, 6),
                        output_projection=_hashed_vector(
                            f"{basis}|output|{turn.content}",
                            config.feature_schema.output_projection_dim,
                        ),
                        source_turn_id=turn.turn_id,
                        source_speaker=turn.speaker,
                    )
                )

    return FeatureHarvest(
        sample_id=sample.sample_id,
        boundary_id=sample.boundary.boundary_id,
        logical_context_tokens=sample.logical_context_tokens,
        physical_context_tokens=sample.physical_context_tokens,
        feature_granularity=config.feature_schema.granularity,
        tap_point=config.feature_schema.tap_point,
        query_projection_dim=config.feature_schema.query_projection_dim,
        output_projection_dim=config.feature_schema.output_projection_dim,
        observed_layers=list(PROBE_LAYERS),
        observed_heads=list(PROBE_HEADS),
        observation_count=len(observations),
        observations=observations,
    )


def write_feature_harvest(harvest: FeatureHarvest, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(harvest.to_serializable(), indent=2) + "\n", encoding="utf-8")
    return output_path
