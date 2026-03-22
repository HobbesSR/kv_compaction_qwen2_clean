from __future__ import annotations

from pathlib import Path

from kv_compaction_clean.behavioral_eval import run_behavioral_evaluation, write_behavioral_result
from kv_compaction_clean.config import load_config
from kv_compaction_clean.context_loader import load_context_sample
from kv_compaction_clean.service_demo import (
    build_service_demo_session,
    format_progress_event,
    write_service_demo_summary,
)


DEFAULT_CONFIG_PATH = "configs/qwen25_smoke/qwen2_5_3b.yaml"


def run_smoke_eval() -> None:
    config = load_config(DEFAULT_CONFIG_PATH)
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


def run_service_demo() -> None:
    config = load_config(DEFAULT_CONFIG_PATH)
    sample = load_context_sample(config)

    def on_progress(event: dict[str, object]) -> None:
        print(format_progress_event(event), flush=True)

    session = build_service_demo_session(
        sample,
        config,
        keys_per_head=6,
        progress_callback=on_progress,
    )
    try:
        summary_path = Path("artifacts/qwen25_smoke/service_demo_summary.json")
        write_service_demo_summary(session.summary, summary_path)
        print(f"Compaction ready: {summary_path}")
        print("Commands: /compact <prompt>, /full <prompt>, /status, /quit")
        while True:
            raw = input("> ").strip()
            if not raw:
                continue
            if raw in {"/quit", "quit", "exit"}:
                break
            if raw == "/status":
                print(session.summary.to_serializable())
                continue
            compacted = True
            prompt_text = raw
            if raw.startswith("/full "):
                compacted = False
                prompt_text = raw[len("/full ") :]
            elif raw.startswith("/compact "):
                compacted = True
                prompt_text = raw[len("/compact ") :]
            if not prompt_text:
                print("Prompt text is required.")
                continue
            answer, runtime = session.answer(prompt_text, compacted=compacted)
            mode = "compact" if compacted else "full"
            print(f"[{mode} {runtime:.3f}s] {answer}")
    finally:
        session.close()


def export_example_summaries() -> None:
    import json

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
