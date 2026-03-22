from __future__ import annotations

from pathlib import Path

from kv_compaction_clean.behavioral_eval import run_behavioral_evaluation, write_behavioral_result
from kv_compaction_clean.config import load_config
from kv_compaction_clean.context_loader import load_context_sample


def main() -> None:
    config = load_config("configs/qwen25_smoke/qwen2_5_3b.yaml")
    sample = load_context_sample(config)
    result = run_behavioral_evaluation(
        sample,
        config,
        keys_per_head=6,
        prompt_set="phase2_clean",
    )
    output_path = Path("artifacts/qwen25_smoke/behavioral_eval_phase2_clean_k6.json")
    write_behavioral_result(result, output_path)
    print(output_path)


if __name__ == "__main__":
    main()
