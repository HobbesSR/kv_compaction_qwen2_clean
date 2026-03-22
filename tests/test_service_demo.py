from __future__ import annotations

from kv_compaction_clean.service_demo import ServiceDemoSummary, format_progress_event


def test_format_progress_event_prefill_and_capture() -> None:
    prefill = format_progress_event(
        {
            "stage": "prefill",
            "processed_token_count": 1024,
            "prefix_token_count": 4096,
        }
    )
    capture = format_progress_event(
        {
            "stage": "capture",
            "processed_token_count": 2048,
            "prefix_token_count": 4096,
            "monitored_observation_count": 84,
            "monitored_query_sample_count": 84,
        }
    )

    assert prefill == "[prefill] 1024/4096 tokens (25.0%)"
    assert capture == "[capture] 2048/4096 tokens (50.0%) obs=84 queries=84"


def test_service_demo_summary_serializes() -> None:
    summary = ServiceDemoSummary(
        sample_id="sample",
        boundary_id="boundary",
        keys_per_head=6,
        compacted_head_count=12,
        effective_compact_tokens=72,
        prefix_token_count=7168,
        preserved_tail_tokens=1024,
        capture_token_count=29,
        monitored_observation_count=348,
        monitored_query_sample_count=348,
    )

    payload = summary.to_serializable()

    assert payload["compacted_head_count"] == 12
    assert payload["effective_compact_tokens"] == 72
