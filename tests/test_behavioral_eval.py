from __future__ import annotations

from kv_compaction_clean.behavioral_eval import build_prompt_set, evaluate_run
from kv_compaction_clean.data_types import BehavioralPrompt, FactExpectation


def test_build_prompt_set_phase2_clean_has_expected_labels() -> None:
    prompts = build_prompt_set("phase2_clean")

    assert len(prompts) == 6
    assert prompts[0].label == "same_task_handoff_and_rollback"
    assert prompts[-1].label == "branch_switch_tool_references"


def test_evaluate_run_marks_missing_and_hallucinated_facts() -> None:
    prompt = BehavioralPrompt(
        label="prompt",
        category="branch_switch",
        prompt_text="Answer directly.",
        required_facts=[
            FactExpectation("dock_three", ["dock three", "dock 3"]),
            FactExpectation("supplier_phones", ["supplier phone numbers", "supplier phone"]),
        ],
        forbidden_markers=["dock four"],
        target_head_labels=["12:7"],
    )

    result = evaluate_run(
        prompt=prompt,
        generated_text="Dock three needs the relay harness note, but dock four is also mentioned.",
        runtime_seconds=1.25,
        reference_text="Dock three and supplier phone numbers.",
        reference_hits=["dock_three", "supplier_phones"],
    )

    assert result.required_fact_labels_hit == ["dock_three"]
    assert result.missing_required_fact_labels == ["supplier_phones"]
    assert result.hallucination_flags == ["dock four"]
    assert result.reference_missing_fact_labels == ["supplier_phones"]
    assert result.reference_unigram_f1 is not None
