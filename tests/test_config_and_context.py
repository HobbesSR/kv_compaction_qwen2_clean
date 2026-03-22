from pathlib import Path

from kv_compaction_clean.config import load_config
from kv_compaction_clean.context_loader import load_context_sample


def test_qwen25_smoke_config_loads() -> None:
    config = load_config(Path("configs/qwen25_smoke/qwen2_5_3b.yaml"))

    assert config.experiment.name == "qwen25_smoke"
    assert config.model.huggingface_id == "Qwen/Qwen2.5-3B"
    assert config.data.dataset == "local_placeholder"


def test_qwen25_smoke_context_loads() -> None:
    config = load_config(Path("configs/qwen25_smoke/qwen2_5_3b.yaml"))
    sample = load_context_sample(config)

    assert sample.sample_id == "local_smoke_branch_switch_v0"
    assert sample.logical_context_tokens == 8192
    assert sample.boundary.prefix_token_count == 7168
    assert sample.boundary.target_context_tokens_after_compaction == 1741
    assert [chunk.chunk_id for chunk in sample.chunks] == ["chunk_0", "chunk_1"]
