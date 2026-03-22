from __future__ import annotations

from dataclasses import asdict, dataclass
import gc
from importlib.util import find_spec
import json
import math
from pathlib import Path
from typing import Any

from kv_compaction_clean.data_types import LoadedContextSample, SmokeTestConfig


PROBE_LAYERS = (4, 12, 20, 28)
PROBE_HEADS = (0, 3, 7)
SUPPORTED_DTYPES = {
    "float32": "float32",
    "float16": "float16",
    "bfloat16": "bfloat16",
}


@dataclass
class RuntimeDependencyStatus:
    torch_available: bool
    transformers_available: bool
    accelerate_available: bool
    ready_for_model_load: bool
    missing_packages: list[str]

    def to_serializable(self) -> dict[str, object]:
        return asdict(self)


@dataclass
class ModelRuntimePlan:
    sample_id: str
    boundary_id: str
    provider: str
    model_name: str
    huggingface_id: str
    tokenizer_name: str
    device: str
    dtype: str
    attn_implementation: str
    trust_remote_code: bool
    local_files_only: bool
    logical_context_tokens: int
    physical_context_tokens: int
    max_context_tokens: int
    prefix_token_count: int
    preserved_tail_tokens: int
    prefill_chunk_size: int
    probe_max_tokens: int
    expected_prefill_chunks: int
    capture_layers: list[int]
    capture_heads: list[int]
    tap_point: str
    chat_transcript_turns: int
    transcript_preview: str
    branch_switch_labels: list[str]
    dependency_status: RuntimeDependencyStatus

    def to_serializable(self) -> dict[str, object]:
        return asdict(self)


def _module_available(module_name: str) -> bool:
    return find_spec(module_name) is not None


def detect_runtime_dependencies() -> RuntimeDependencyStatus:
    torch_available = _module_available("torch")
    transformers_available = _module_available("transformers")
    accelerate_available = _module_available("accelerate")

    missing_packages = []
    if not torch_available:
        missing_packages.append("torch")
    if not transformers_available:
        missing_packages.append("transformers")

    return RuntimeDependencyStatus(
        torch_available=torch_available,
        transformers_available=transformers_available,
        accelerate_available=accelerate_available,
        ready_for_model_load=not missing_packages,
        missing_packages=missing_packages,
    )


def _normalize_role(speaker: str) -> str:
    if speaker in {"system", "user", "assistant"}:
        return speaker
    return "tool"


def build_teacher_forced_transcript(sample: LoadedContextSample) -> str:
    blocks = []
    for turn in sample.turns:
        role = _normalize_role(turn.speaker).upper()
        blocks.append(f"{role} [{turn.turn_id}]\n{turn.content}")
    return "\n\n".join(blocks)


def materialize_long_context_ids(
    sample: LoadedContextSample,
    tokenizer,
) -> tuple[list[int], list[tuple[int, int, str, str]]]:
    token_ids: list[int] = []
    spans: list[tuple[int, int, str, str]] = []

    for turn in sample.turns:
        role = _normalize_role(turn.speaker).upper()
        base_ids = tokenizer.encode(
            f"{role} [{turn.turn_id}]\n{turn.content}\n\n",
            add_special_tokens=False,
        )
        if not base_ids:
            raise ValueError(f"Tokenizer produced no ids for turn {turn.turn_id}.")

        repeat_count = (turn.token_count + len(base_ids) - 1) // len(base_ids)
        materialized = (base_ids * repeat_count)[: turn.token_count]
        start = len(token_ids)
        token_ids.extend(materialized)
        spans.append((start, len(token_ids), turn.turn_id, turn.speaker))

    return token_ids, spans


def build_hf_load_kwargs(config: SmokeTestConfig) -> dict[str, Any]:
    if config.model.dtype not in SUPPORTED_DTYPES:
        raise ValueError(
            f"Unsupported dtype {config.model.dtype!r}. "
            f"Expected one of {sorted(SUPPORTED_DTYPES)}."
        )

    dependency_status = detect_runtime_dependencies()
    load_kwargs = {
        "trust_remote_code": config.model.trust_remote_code,
        "local_files_only": config.model.local_files_only,
        "attn_implementation": config.model.attn_implementation,
        "torch_dtype": config.model.dtype,
    }
    if dependency_status.accelerate_available:
        load_kwargs["low_cpu_mem_usage"] = True
    return load_kwargs


def build_model_runtime_plan(sample: LoadedContextSample, config: SmokeTestConfig) -> ModelRuntimePlan:
    dependency_status = detect_runtime_dependencies()
    transcript = build_teacher_forced_transcript(sample)
    expected_prefill_chunks = math.ceil(sample.logical_context_tokens / config.model.prefill_chunk_size)

    return ModelRuntimePlan(
        sample_id=sample.sample_id,
        boundary_id=sample.boundary.boundary_id,
        provider=config.model.provider,
        model_name=config.model.name,
        huggingface_id=config.model.huggingface_id,
        tokenizer_name=config.model.tokenizer_name,
        device=config.model.device,
        dtype=config.model.dtype,
        attn_implementation=config.model.attn_implementation,
        trust_remote_code=config.model.trust_remote_code,
        local_files_only=config.model.local_files_only,
        logical_context_tokens=sample.logical_context_tokens,
        physical_context_tokens=sample.physical_context_tokens,
        max_context_tokens=config.model.max_context_tokens,
        prefix_token_count=sample.boundary.prefix_token_count,
        preserved_tail_tokens=sample.boundary.preserved_tail_tokens,
        prefill_chunk_size=config.model.prefill_chunk_size,
        probe_max_tokens=config.model.probe_max_tokens,
        expected_prefill_chunks=expected_prefill_chunks,
        capture_layers=list(PROBE_LAYERS),
        capture_heads=list(PROBE_HEADS),
        tap_point=config.feature_schema.tap_point,
        chat_transcript_turns=len(sample.turns),
        transcript_preview=transcript[:240],
        branch_switch_labels=[
            sample.boundary.primary_prompt_label,
            sample.boundary.alternate_prompt_label,
        ],
        dependency_status=dependency_status,
    )


def load_hf_model_bundle(config: SmokeTestConfig):
    load_kwargs = build_hf_load_kwargs(config)

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "Model runtime dependencies are missing. Install torch and transformers "
            "before running the clean smoke path."
        ) from exc

    dtype = getattr(torch, load_kwargs.pop("torch_dtype"))
    tokenizer = AutoTokenizer.from_pretrained(
        config.model.tokenizer_name,
        trust_remote_code=config.model.trust_remote_code,
        local_files_only=config.model.local_files_only,
    )
    model = AutoModelForCausalLM.from_pretrained(
        config.model.huggingface_id,
        torch_dtype=dtype,
        **load_kwargs,
    )
    model.eval()
    model = model.to(config.model.device)
    return model, tokenizer


def unload_hf_model_bundle(model) -> None:
    try:
        import torch
    except ImportError:  # pragma: no cover
        return

    if model is None:
        return

    model.to("cpu")
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass


def write_model_runtime_plan(plan: ModelRuntimePlan, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(plan.to_serializable(), indent=2) + "\n", encoding="utf-8")
    return output_path
