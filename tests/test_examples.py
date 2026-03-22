from __future__ import annotations

import json
from pathlib import Path


def test_qwen25_examples_exist_and_have_expected_shape() -> None:
    behavioral = json.loads(
        Path("examples/qwen25_smoke/behavioral_eval_summary.json").read_text(encoding="utf-8")
    )
    service = json.loads(
        Path("examples/qwen25_smoke/service_demo_summary.json").read_text(encoding="utf-8")
    )

    assert behavioral["prompt_set"] == "phase2_clean"
    assert behavioral["sketch"]["path"] == "prototype_bank"
    assert behavioral["control"]["path"] == "teacher_forced_subsample"
    assert service["effective_compact_tokens"] == behavioral["sketch"]["effective_compact_tokens"]
    assert service["compacted_head_count"] == behavioral["sketch"]["compacted_head_count"]
