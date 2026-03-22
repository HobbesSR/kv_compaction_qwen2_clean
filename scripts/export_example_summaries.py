from __future__ import annotations

import json
from pathlib import Path


def main() -> None:
    artifact_root = Path("artifacts/qwen25_smoke")
    example_root = Path("examples/qwen25_smoke")
    example_root.mkdir(parents=True, exist_ok=True)

    behavioral_path = artifact_root / "behavioral_eval_phase2_clean_k6.json"
    behavioral = json.loads(behavioral_path.read_text(encoding="utf-8"))
    behavioral_summary = {
        "sample_id": behavioral["sample_id"],
        "boundary_id": behavioral["boundary_id"],
        "prompt_set": behavioral["prompt_set"],
        "keys_per_head": behavioral["keys_per_head"],
        "key_selection_method": behavioral["key_selection_method"],
        "prompt_labels": behavioral["prompt_labels"],
        "reference": {
            key: behavioral["reference"][key]
            for key in (
                "runtime_seconds",
                "preserved_central_detail_count",
                "omitted_central_detail_count",
                "hallucination_run_count",
                "effective_compact_tokens",
            )
        },
        "sketch": {
            key: behavioral["sketch"][key]
            for key in (
                "path",
                "runtime_seconds",
                "preserved_central_detail_count",
                "omitted_central_detail_count",
                "hallucination_run_count",
                "effective_compact_tokens",
                "compacted_head_count",
            )
        },
        "control": {
            key: behavioral["control"][key]
            for key in (
                "path",
                "runtime_seconds",
                "preserved_central_detail_count",
                "omitted_central_detail_count",
                "hallucination_run_count",
                "effective_compact_tokens",
                "compacted_head_count",
            )
        },
    }
    (example_root / "behavioral_eval_summary.json").write_text(
        json.dumps(behavioral_summary, indent=2) + "\n",
        encoding="utf-8",
    )

    service_summary = json.loads((artifact_root / "service_demo_summary.json").read_text(encoding="utf-8"))
    (example_root / "service_demo_summary.json").write_text(
        json.dumps(service_summary, indent=2) + "\n",
        encoding="utf-8",
    )

    print(example_root)


if __name__ == "__main__":
    main()
