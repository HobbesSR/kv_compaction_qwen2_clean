"""Clean smoke-test implementation surface for KV compaction."""

from kv_compaction_clean.config import load_config
from kv_compaction_clean.context_loader import load_context_sample

__all__ = ["load_config", "load_context_sample"]
