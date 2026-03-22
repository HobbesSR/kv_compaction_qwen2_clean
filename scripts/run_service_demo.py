from __future__ import annotations

from pathlib import Path

from kv_compaction_clean.config import load_config
from kv_compaction_clean.context_loader import load_context_sample
from kv_compaction_clean.service_demo import (
    build_service_demo_session,
    format_progress_event,
    write_service_demo_summary,
)


def main() -> None:
    config = load_config("configs/qwen25_smoke/qwen2_5_3b.yaml")
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


if __name__ == "__main__":
    main()
