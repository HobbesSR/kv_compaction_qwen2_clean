from kv_compaction_clean.roadmap import PUBLIC_TRACKS


def test_public_tracks_are_declared() -> None:
    assert PUBLIC_TRACKS == (
        "qwen25_smoke",
        "service_demo",
    )
