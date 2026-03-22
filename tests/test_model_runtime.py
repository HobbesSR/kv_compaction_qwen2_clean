from __future__ import annotations

import json
from pathlib import Path

from kv_compaction_clean.config import load_config
from kv_compaction_clean.context_loader import load_context_sample
from kv_compaction_clean.model_runtime import (
    build_hf_load_kwargs,
    build_model_runtime_plan,
    build_teacher_forced_transcript,
    detect_runtime_dependencies,
    write_model_runtime_plan,
)


def test_build_teacher_forced_transcript_includes_turns() -> None:
    config = load_config("configs/qwen25_smoke/qwen2_5_3b.yaml")
    sample = load_context_sample(config)

    transcript = build_teacher_forced_transcript(sample)

    assert "SYSTEM [turn_0]" in transcript
    assert "TOOL [turn_7]" in transcript
    assert "night shift has to roll back only dock three" in transcript


def test_build_model_runtime_plan_matches_config() -> None:
    config = load_config("configs/qwen25_smoke/qwen2_5_3b.yaml")
    sample = load_context_sample(config)

    plan = build_model_runtime_plan(sample, config)

    assert plan.provider == "huggingface"
    assert plan.huggingface_id == "Qwen/Qwen2.5-3B"
    assert plan.tokenizer_name == "Qwen/Qwen2.5-3B"
    assert plan.prefill_chunk_size == 2048
    assert plan.probe_max_tokens == 256
    assert plan.expected_prefill_chunks == 4
    assert plan.capture_layers == [4, 12, 20, 28]
    assert plan.capture_heads == [0, 3, 7]
    assert plan.branch_switch_labels == [
        "continue_original_task",
        "retrieve_different_details",
    ]


def test_build_hf_load_kwargs_uses_runtime_settings() -> None:
    config = load_config("configs/qwen25_smoke/qwen2_5_3b.yaml")
    status = detect_runtime_dependencies()

    kwargs = build_hf_load_kwargs(config)

    assert kwargs["attn_implementation"] == "sdpa"
    assert kwargs["torch_dtype"] == "bfloat16"
    assert kwargs["trust_remote_code"] is False
    assert kwargs["local_files_only"] is False
    assert kwargs.get("low_cpu_mem_usage", False) is status.accelerate_available


def test_detect_runtime_dependencies_reports_status_types() -> None:
    status = detect_runtime_dependencies()

    assert isinstance(status.torch_available, bool)
    assert isinstance(status.transformers_available, bool)
    assert isinstance(status.accelerate_available, bool)
    assert isinstance(status.ready_for_model_load, bool)
    assert isinstance(status.missing_packages, list)


def test_write_model_runtime_plan_serializes_json(tmp_path: Path) -> None:
    config = load_config("configs/qwen25_smoke/qwen2_5_3b.yaml")
    sample = load_context_sample(config)
    plan = build_model_runtime_plan(sample, config)
    output_path = tmp_path / "model_runtime_plan.json"

    write_model_runtime_plan(plan, output_path)

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["model_name"] == "qwen2.5-3b"
    assert payload["dependency_status"]["ready_for_model_load"] in {True, False}
    assert payload["expected_prefill_chunks"] == 4
