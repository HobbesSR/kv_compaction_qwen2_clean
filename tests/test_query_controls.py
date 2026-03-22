from __future__ import annotations

from kv_compaction_clean.data_types import QuerySample, QuerySampleBank
from kv_compaction_clean.query_controls import extract_teacher_forced_subsample_control


def test_extract_teacher_forced_subsample_control_ranks_by_mass() -> None:
    query_bank = QuerySampleBank(
        sample_id="sample",
        boundary_id="boundary",
        query_dim=2,
        sample_count=3,
        samples=[
            QuerySample(
                query_id="q0",
                layer=4,
                head=0,
                token_index=10,
                prefix_mass_share=0.2,
                raw_prefix_mass=2.0,
                query_projection=[1.0, 0.0],
                raw_query_vector=[1.0, 0.0],
                source_turn_id="turn_0",
                source_speaker="user",
            ),
            QuerySample(
                query_id="q1",
                layer=4,
                head=0,
                token_index=20,
                prefix_mass_share=0.5,
                raw_prefix_mass=10.0,
                query_projection=[0.0, 1.0],
                raw_query_vector=[0.0, 1.0],
                source_turn_id="turn_1",
                source_speaker="tool",
            ),
            QuerySample(
                query_id="q2",
                layer=12,
                head=3,
                token_index=30,
                prefix_mass_share=0.4,
                raw_prefix_mass=12.0,
                query_projection=[0.5, 0.5],
                raw_query_vector=[0.5, 0.5],
                source_turn_id="turn_2",
                source_speaker="assistant",
            ),
        ],
    )

    control = extract_teacher_forced_subsample_control(query_bank, max_entries=2)

    assert control.source == "teacher_forced_subsample"
    assert len(control.selected_entries) == 2
    assert control.selected_entries[0].prototype_id == "q2"
    assert control.selected_entries[1].prototype_id == "q1"
