from __future__ import annotations

from collections import Counter
from contextlib import nullcontext
from dataclasses import asdict, replace
import json
import math
import re
import time
from pathlib import Path

from kv_compaction_clean.boundary_collection import collect_teacher_forced_boundary_collection
from kv_compaction_clean.coreset import extract_query_coreset
from kv_compaction_clean.data_types import (
    BehavioralEvalResult,
    BehavioralPathResult,
    BehavioralPrompt,
    BehavioralRunResult,
    FactExpectation,
)
from kv_compaction_clean.model_runtime import load_hf_model_bundle, materialize_long_context_ids, unload_hf_model_bundle
from kv_compaction_clean.prototype_bank import build_state_from_observations
from kv_compaction_clean.query_controls import extract_teacher_forced_subsample_control
from kv_compaction_clean.runtime_compaction import (
    BETA_REGULARIZATION,
    BETA_SOLVER,
    TRAIN_FRACTION,
    VALUE_REGULARIZATION,
    build_path_runtime,
    patched_compaction_attention,
)


MAX_NEW_TOKENS = 64
EAGER_PREFILL_ATTENTION_BUDGET_BYTES = 64 * 1024 * 1024
DEFAULT_PROMPT_SET = "phase2_clean"
VALIDATED_TARGET_LAYERS = (4, 12, 20, 28)
VALIDATED_TARGET_HEADS = (0, 3, 7)
VALIDATED_TARGET_LAYER_HEADS = tuple((layer, head) for layer in VALIDATED_TARGET_LAYERS for head in VALIDATED_TARGET_HEADS)
UNSUPPORTED_MARKERS = (
    "dock four",
    "dock 4",
    "dock five",
    "dock 5",
    "south dock",
    "west dock",
    "database migration",
    "monday cutover",
    "tuesday cutover",
    "friday cutover",
)


def _build_prompts() -> list[BehavioralPrompt]:
    return [
        BehavioralPrompt(
            label="same_task_handoff_and_rollback",
            category="same_task",
            prompt_text=(
                "Answer directly without repeating the question. Provide four short bullets covering: "
                "(1) the operator handoff checklist, "
                "(2) the rollback order if only dock three must roll back, "
                "(3) the controller swap note, and "
                "(4) the decision gates for moving from dry run to live traffic."
            ),
            required_facts=[
                FactExpectation("handoff_checklist", ["handoff checklist", "checklist"]),
                FactExpectation("dock_three_rollback", ["dock three", "dock 3"]),
                FactExpectation("rollback_order", ["dock two", "dock one"]),
                FactExpectation("controller_swap", ["controller", "swap"]),
                FactExpectation("decision_gates", ["decision gate", "dry run", "live traffic"]),
            ],
            forbidden_markers=list(UNSUPPORTED_MARKERS),
            target_head_labels=["12:3", "20:0", "20:7", "28:7"],
        ),
        BehavioralPrompt(
            label="same_task_rollout_structure",
            category="same_task",
            prompt_text=(
                "Answer directly without repeating the question. Give three short bullets covering: "
                "(1) the rollout structure, "
                "(2) the controller replacement sequence, and "
                "(3) the dry-run-to-live decision gates."
            ),
            required_facts=[
                FactExpectation("phased_rollout", ["phased rollout", "phased approach", "rollout is phased"]),
                FactExpectation("controller_replacement_sequence", ["controller replacement sequence", "replacement order"]),
                FactExpectation("dry_run_to_live", ["dry run", "live traffic", "decision gate"]),
            ],
            forbidden_markers=list(UNSUPPORTED_MARKERS),
            target_head_labels=["20:0", "20:7"],
        ),
        BehavioralPrompt(
            label="same_task_appendix_guardrails",
            category="same_task",
            prompt_text=(
                "Answer directly without repeating the question. Name the late details that the operator handoff must "
                "still preserve even if the team stays focused on the rollout checklist."
            ),
            required_facts=[
                FactExpectation("relay_harness", ["relay harness"]),
                FactExpectation("supplier_phones", ["supplier phone numbers", "supplier phone"]),
                FactExpectation("cage_inventory", ["cage inventory"]),
                FactExpectation("shift_lead_names", ["shift lead names", "shift leads"]),
            ],
            forbidden_markers=list(UNSUPPORTED_MARKERS),
            target_head_labels=["12:7", "20:0", "28:7"],
        ),
        BehavioralPrompt(
            label="branch_switch_harness_and_appendix",
            category="branch_switch",
            prompt_text=(
                "Answer directly without repeating the question. State which dock requires a different relay harness, "
                "where that note was recorded, and list three other late-appendix details that should not be lost."
            ),
            required_facts=[
                FactExpectation("relay_harness", ["relay harness"]),
                FactExpectation("dock_three", ["dock three", "dock 3"]),
                FactExpectation("late_appendix", ["late appendix", "appendix"]),
                FactExpectation("supplier_phones", ["supplier phone numbers", "supplier phone"]),
                FactExpectation("cage_inventory", ["cage inventory"]),
                FactExpectation("shift_lead_names", ["shift lead names", "shift leads"]),
            ],
            forbidden_markers=list(UNSUPPORTED_MARKERS),
            target_head_labels=["12:7", "20:0", "28:7"],
        ),
        BehavioralPrompt(
            label="branch_switch_appendix_only",
            category="branch_switch",
            prompt_text=(
                "Answer directly without repeating the question. Ignore rollout steps and list the three late-appendix "
                "details, besides the relay harness note, that should be retained."
            ),
            required_facts=[
                FactExpectation("supplier_phones", ["supplier phone numbers", "supplier phone"]),
                FactExpectation("cage_inventory", ["cage inventory"]),
                FactExpectation("shift_lead_names", ["shift lead names", "shift leads"]),
            ],
            forbidden_markers=list(UNSUPPORTED_MARKERS),
            target_head_labels=["12:7", "28:7"],
        ),
        BehavioralPrompt(
            label="branch_switch_tool_references",
            category="branch_switch",
            prompt_text=(
                "Answer directly without repeating the question. Give two short bullets: "
                "(1) which dock differs from docks one and two, and "
                "(2) which tool-side factual references were recorded with that hardware note."
            ),
            required_facts=[
                FactExpectation("dock_three", ["dock three", "dock 3"]),
                FactExpectation("supplier_phones", ["supplier phone numbers", "supplier phone"]),
                FactExpectation("cage_inventory", ["cage inventory"]),
                FactExpectation("shift_lead_names", ["shift lead names", "shift leads"]),
            ],
            forbidden_markers=list(UNSUPPORTED_MARKERS),
            target_head_labels=["12:7", "20:0", "28:7"],
        ),
    ]


PROMPT_SET_LABELS = {
    "phase2_clean": [
        "same_task_handoff_and_rollback",
        "same_task_rollout_structure",
        "same_task_appendix_guardrails",
        "branch_switch_harness_and_appendix",
        "branch_switch_appendix_only",
        "branch_switch_tool_references",
    ]
}


def build_prompt_set(prompt_set: str = DEFAULT_PROMPT_SET) -> list[BehavioralPrompt]:
    prompts_by_label = {prompt.label: prompt for prompt in _build_prompts()}
    try:
        labels = PROMPT_SET_LABELS[prompt_set]
    except KeyError as exc:
        raise ValueError(f"Unsupported prompt set {prompt_set!r}.") from exc
    return [prompts_by_label[label] for label in labels]


def _normalise_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _cleanup_generated_text(text: str) -> str:
    cleaned = text.strip()
    role_marker = re.search(r"\b(?:USER|ASSISTANT|SYSTEM|TOOL) \[[^\]]+\]", cleaned)
    if role_marker is not None:
        cleaned = cleaned[: role_marker.start()].strip()
    return cleaned


def _keyword_recall(text: str, keyword_groups: list[list[str]]) -> tuple[int, int, float]:
    lowered = _normalise_text(text)
    hits = 0
    for group in keyword_groups:
        if any(keyword in lowered for keyword in group):
            hits += 1
    total = len(keyword_groups)
    recall = round((hits / total) if total else 0.0, 6)
    return hits, total, recall


def _token_counts(text: str) -> Counter[str]:
    return Counter(re.findall(r"[a-z0-9']+", _normalise_text(text)))


def _unigram_f1(reference: str, candidate: str) -> float:
    reference_counts = _token_counts(reference)
    candidate_counts = _token_counts(candidate)
    overlap = sum(min(reference_counts[token], candidate_counts[token]) for token in reference_counts)
    if overlap == 0:
        return 0.0
    precision = overlap / max(sum(candidate_counts.values()), 1)
    recall = overlap / max(sum(reference_counts.values()), 1)
    return round((2.0 * precision * recall) / max(precision + recall, 1e-12), 6)


def _fact_labels_hit(text: str, facts: list[FactExpectation]) -> list[str]:
    lowered = _normalise_text(text)
    return [fact.label for fact in facts if any(keyword in lowered for keyword in fact.keywords)]


def _hallucination_flags(text: str, forbidden_markers: list[str]) -> list[str]:
    lowered = _normalise_text(text)
    return sorted(marker for marker in forbidden_markers if marker in lowered)


def _divergence_summary(
    reference_hits: list[str],
    candidate_hits: list[str],
    hallucination_flags: list[str],
    reference_unigram_f1: float | None,
) -> str:
    if reference_unigram_f1 is None:
        return "reference run"
    missing_vs_reference = [label for label in reference_hits if label not in candidate_hits]
    extra_vs_reference = [label for label in candidate_hits if label not in reference_hits]
    parts = []
    if missing_vs_reference:
        parts.append(f"misses reference-hit facts: {', '.join(missing_vs_reference)}")
    if extra_vs_reference:
        parts.append(f"adds non-reference facts: {', '.join(extra_vs_reference)}")
    if hallucination_flags:
        parts.append(f"unsupported markers: {', '.join(hallucination_flags)}")
    if not parts:
        if reference_unigram_f1 >= 0.95:
            return "matches reference closely"
        return "preserves required facts with wording differences"
    return "; ".join(parts)


def evaluate_run(
    prompt: BehavioralPrompt,
    generated_text: str,
    runtime_seconds: float,
    reference_text: str | None = None,
    reference_hits: list[str] | None = None,
) -> BehavioralRunResult:
    required_fact_labels_hit = sorted(_fact_labels_hit(generated_text, prompt.required_facts))
    required_fact_labels_hit_set = set(required_fact_labels_hit)
    required_fact_labels = [fact.label for fact in prompt.required_facts]
    missing_required_fact_labels = [label for label in required_fact_labels if label not in required_fact_labels_hit_set]

    central_labels = [fact.label for fact in prompt.required_facts if fact.central]
    central_fact_labels_hit = [label for label in required_fact_labels_hit if label in set(central_labels)]
    missing_central_fact_labels = [label for label in central_labels if label not in required_fact_labels_hit_set]
    central_detail_preserved = not missing_central_fact_labels
    hallucination_flags = _hallucination_flags(generated_text, prompt.forbidden_markers)

    keyword_groups = [fact.keywords for fact in prompt.required_facts]
    keyword_hits, keyword_total, keyword_recall = _keyword_recall(generated_text, keyword_groups)
    reference_unigram = _unigram_f1(reference_text, generated_text) if reference_text is not None else None
    reference_hits = reference_hits or []
    reference_missing_fact_labels = [label for label in reference_hits if label not in required_fact_labels_hit_set]
    reference_extra_fact_labels = [label for label in required_fact_labels_hit if label not in set(reference_hits)]

    return BehavioralRunResult(
        label=prompt.label,
        category=prompt.category,
        prompt_text=prompt.prompt_text,
        target_head_labels=list(prompt.target_head_labels),
        generated_text=generated_text,
        success=True,
        runtime_seconds=runtime_seconds,
        keyword_hits=keyword_hits,
        keyword_total=keyword_total,
        keyword_recall=keyword_recall,
        required_fact_labels_hit=required_fact_labels_hit,
        missing_required_fact_labels=missing_required_fact_labels,
        central_fact_labels_hit=central_fact_labels_hit,
        missing_central_fact_labels=missing_central_fact_labels,
        central_detail_preserved=central_detail_preserved,
        omitted_central_detail=not central_detail_preserved,
        hallucination_flags=hallucination_flags,
        reference_missing_fact_labels=reference_missing_fact_labels,
        reference_extra_fact_labels=reference_extra_fact_labels,
        divergence_summary=_divergence_summary(
            reference_hits=reference_hits,
            candidate_hits=required_fact_labels_hit,
            hallucination_flags=hallucination_flags,
            reference_unigram_f1=reference_unigram,
        ),
        reference_unigram_f1=reference_unigram,
    )


def _build_path_result(
    path: str,
    keys_per_head: int,
    compacted_layers,
    prompt_results: list[BehavioralRunResult],
    runtime_seconds: float,
    prefix_token_count: int,
) -> BehavioralPathResult:
    compacted_heads: list[dict[str, object]] = []
    effective_compact_tokens = 0
    if compacted_layers:
        for layer in sorted(compacted_layers):
            for head in sorted(compacted_layers[layer]):
                runtime = compacted_layers[layer][head]
                effective_compact_tokens += len(runtime.selected_indices)
                compacted_heads.append(
                    {
                        "layer": layer,
                        "head": head,
                        "selected_indices": runtime.selected_indices,
                        "selected_key_count": len(runtime.selected_indices),
                    }
                )

    return BehavioralPathResult(
        path=path,
        keys_per_head=keys_per_head,
        compaction_succeeded=True,
        compacted_head_count=len(compacted_heads),
        compacted_prefix_tokens=prefix_token_count,
        effective_compact_tokens=effective_compact_tokens,
        runtime_seconds=round(runtime_seconds, 6),
        preserved_central_detail_count=sum(run.central_detail_preserved for run in prompt_results),
        omitted_central_detail_count=sum(run.omitted_central_detail for run in prompt_results),
        hallucination_run_count=sum(bool(run.hallucination_flags) for run in prompt_results),
        runs=prompt_results,
        compacted_heads=compacted_heads,
    )


def write_behavioral_result(result: BehavioralEvalResult, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(asdict(result), indent=2) + "\n", encoding="utf-8")
    return output_path


def _assistant_prefill_ids(tokenizer, prompt_text: str) -> list[int]:
    prompt = f"USER [behavior_eval]\n{prompt_text}\n\nASSISTANT [behavior_answer]\n"
    return tokenizer.encode(prompt, add_special_tokens=False)


def _feed_tokens_with_cache(
    model,
    token_ids: list[int],
    device: str,
    chunk_size: int | None = None,
    past_key_values=None,
    start_position: int = 0,
):
    import torch

    if not token_ids:
        return past_key_values

    input_tensor = torch.tensor([token_ids], device=device, dtype=torch.long)
    processed = 0
    chunk_size = chunk_size or len(token_ids)
    last_logits = None

    while processed < len(token_ids):
        chunk_end = min(len(token_ids), processed + chunk_size)
        chunk_ids = input_tensor[:, processed:chunk_end]
        cache_position = torch.arange(start_position + processed, start_position + chunk_end, device=device)
        with torch.inference_mode():
            outputs = model(
                input_ids=chunk_ids,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
                cache_position=cache_position,
            )
        past_key_values = outputs.past_key_values
        last_logits = outputs.logits[:, -1, :]
        processed = chunk_end
    return past_key_values, last_logits


def _torch_dtype_bytes(dtype) -> int:
    import torch

    if dtype in {torch.float16, torch.bfloat16, torch.int16, torch.uint16}:
        return 2
    if dtype in {torch.float32, torch.int32}:
        return 4
    if dtype in {torch.float64, torch.int64}:
        return 8
    return 4


def _bounded_eager_prefill_chunk_size(
    *,
    requested_chunk_size: int,
    context_tokens: int,
    num_attention_heads: int,
    bytes_per_attention_element: int,
) -> int:
    if requested_chunk_size <= 0:
        raise ValueError("requested_chunk_size must be positive.")
    if context_tokens <= 0 or num_attention_heads <= 0 or bytes_per_attention_element <= 0:
        return min(requested_chunk_size, 128)

    budget_elements = EAGER_PREFILL_ATTENTION_BUDGET_BYTES // bytes_per_attention_element
    bounded_chunk_size = budget_elements // max(1, context_tokens * num_attention_heads)
    bounded_chunk_size = max(16, (bounded_chunk_size // 16) * 16)
    return min(requested_chunk_size, bounded_chunk_size)


def _effective_prefill_chunk_size(model, *, requested_chunk_size: int, context_tokens: int) -> int:
    attn_implementation = getattr(model.config, "_attn_implementation", None) or getattr(
        model.config,
        "attn_implementation",
        None,
    )
    if attn_implementation != "eager":
        return requested_chunk_size
    return _bounded_eager_prefill_chunk_size(
        requested_chunk_size=requested_chunk_size,
        context_tokens=context_tokens,
        num_attention_heads=int(getattr(model.config, "num_attention_heads", 0) or 0),
        bytes_per_attention_element=_torch_dtype_bytes(model.dtype),
    )


def _continue_with_prompt(
    model,
    tokenizer,
    prefix_token_ids: list[int],
    tail_token_ids: list[int],
    prompt: BehavioralPrompt,
    device: str,
    prefill_chunk_size: int,
    compacted_layers,
    prefix_token_count: int,
) -> tuple[str, float]:
    import torch

    start_time = time.perf_counter()
    patch_context = patched_compaction_attention(compacted_layers, prefix_token_count) if compacted_layers is not None else nullcontext()

    with patch_context:
        effective_prefix_chunk_size = _effective_prefill_chunk_size(
            model,
            requested_chunk_size=prefill_chunk_size,
            context_tokens=max(len(prefix_token_ids), prefix_token_count),
        )
        prefix_cache = _feed_tokens_with_cache(
            model,
            prefix_token_ids,
            device=device,
            chunk_size=effective_prefix_chunk_size,
        )[0]
        logical_position = prefix_token_count
        effective_tail_chunk_size = _effective_prefill_chunk_size(
            model,
            requested_chunk_size=max(1, min(prefill_chunk_size, 128)),
            context_tokens=max(prefix_token_count + len(tail_token_ids), len(tail_token_ids)),
        )
        prefix_cache = _feed_tokens_with_cache(
            model,
            tail_token_ids,
            device=device,
            chunk_size=effective_tail_chunk_size,
            past_key_values=prefix_cache,
            start_position=logical_position,
        )[0]
        logical_position += len(tail_token_ids)

        prompt_token_ids = _assistant_prefill_ids(tokenizer, prompt.prompt_text)
        prefix_cache, logits = _feed_tokens_with_cache(
            model,
            prompt_token_ids,
            device=device,
            chunk_size=max(1, min(len(prompt_token_ids), 64)),
            past_key_values=prefix_cache,
            start_position=logical_position,
        )
        logical_position += len(prompt_token_ids)

        if logits is None:
            raise ValueError("Prompt continuation failed to produce logits.")

        generated_token_ids: list[int] = []
        next_token = int(torch.argmax(logits, dim=-1).item())
        for _ in range(MAX_NEW_TOKENS):
            if tokenizer.eos_token_id is not None and next_token == tokenizer.eos_token_id:
                break
            generated_token_ids.append(next_token)
            token_tensor = torch.tensor([[next_token]], device=device, dtype=torch.long)
            cache_position = torch.tensor([logical_position], device=device, dtype=torch.long)
            with torch.inference_mode():
                outputs = model(
                    input_ids=token_tensor,
                    past_key_values=prefix_cache,
                    use_cache=True,
                    return_dict=True,
                    cache_position=cache_position,
                )
            prefix_cache = outputs.past_key_values
            logits = outputs.logits[:, -1, :]
            next_token = int(torch.argmax(logits, dim=-1).item())
            logical_position += 1

    generated_text = _cleanup_generated_text(tokenizer.decode(generated_token_ids, skip_special_tokens=True).strip())
    return generated_text, round(time.perf_counter() - start_time, 6)


def run_behavioral_evaluation(
    sample,
    config,
    *,
    keys_per_head: int,
    prompt_set: str = DEFAULT_PROMPT_SET,
    key_selection_method: str = "highest_attention",
) -> BehavioralEvalResult:
    prompts = build_prompt_set(prompt_set)
    boundary_config = replace(config, model=replace(config.model, attn_implementation="eager"))
    model, tokenizer = load_hf_model_bundle(boundary_config)
    try:
        bundle = collect_teacher_forced_boundary_collection(sample, boundary_config, model=model, tokenizer=tokenizer)
        state = build_state_from_observations(boundary_config, bundle.harvest.observations)
        sketch_source = extract_query_coreset(sample.sample_id, sample.boundary.boundary_id, state, boundary_config)
        control_query_source = extract_teacher_forced_subsample_control(
            bundle.query_bank,
            max_entries=len(sketch_source.selected_entries),
        )
        sketch_selection, sketch_layers = build_path_runtime(
            sample.sample_id,
            sample.boundary.boundary_id,
            sketch_source.source,
            keys_per_head,
            bundle,
            sketch_source,
            target_layers=VALIDATED_TARGET_LAYERS,
            target_heads=VALIDATED_TARGET_HEADS,
            target_layer_heads=VALIDATED_TARGET_LAYER_HEADS,
            compute_device=boundary_config.model.device,
            key_selection_method=key_selection_method,
        )
        control_selection, control_layers = build_path_runtime(
            sample.sample_id,
            sample.boundary.boundary_id,
            control_query_source.source,
            keys_per_head,
            bundle,
            control_query_source,
            target_layers=VALIDATED_TARGET_LAYERS,
            target_heads=VALIDATED_TARGET_HEADS,
            target_layer_heads=VALIDATED_TARGET_LAYER_HEADS,
            compute_device=boundary_config.model.device,
            key_selection_method=key_selection_method,
        )

        token_ids, _ = materialize_long_context_ids(sample, tokenizer)
        prefix_token_ids = token_ids[: sample.boundary.prefix_token_count]
        tail_token_ids = token_ids[sample.boundary.prefix_token_count :]

        reference_runs: list[BehavioralRunResult] = []
        sketch_runs: list[BehavioralRunResult] = []
        control_runs: list[BehavioralRunResult] = []
        reference_total_runtime = 0.0
        sketch_total_runtime = 0.0
        control_total_runtime = 0.0

        for prompt in prompts:
            reference_text, reference_runtime = _continue_with_prompt(
                model,
                tokenizer,
                prefix_token_ids,
                tail_token_ids,
                prompt,
                device=boundary_config.model.device,
                prefill_chunk_size=boundary_config.model.prefill_chunk_size,
                compacted_layers=None,
                prefix_token_count=sample.boundary.prefix_token_count,
            )
            reference_total_runtime += reference_runtime
            reference_run = evaluate_run(
                prompt=prompt,
                generated_text=reference_text,
                runtime_seconds=reference_runtime,
            )
            reference_runs.append(reference_run)

            sketch_text, sketch_runtime = _continue_with_prompt(
                model,
                tokenizer,
                prefix_token_ids,
                tail_token_ids,
                prompt,
                device=boundary_config.model.device,
                prefill_chunk_size=boundary_config.model.prefill_chunk_size,
                compacted_layers=sketch_layers,
                prefix_token_count=sample.boundary.prefix_token_count,
            )
            sketch_total_runtime += sketch_runtime
            sketch_runs.append(
                evaluate_run(
                    prompt=prompt,
                    generated_text=sketch_text,
                    runtime_seconds=sketch_runtime,
                    reference_text=reference_text,
                    reference_hits=reference_run.required_fact_labels_hit,
                )
            )

            control_text, control_runtime = _continue_with_prompt(
                model,
                tokenizer,
                prefix_token_ids,
                tail_token_ids,
                prompt,
                device=boundary_config.model.device,
                prefill_chunk_size=boundary_config.model.prefill_chunk_size,
                compacted_layers=control_layers,
                prefix_token_count=sample.boundary.prefix_token_count,
            )
            control_total_runtime += control_runtime
            control_runs.append(
                evaluate_run(
                    prompt=prompt,
                    generated_text=control_text,
                    runtime_seconds=control_runtime,
                    reference_text=reference_text,
                    reference_hits=reference_run.required_fact_labels_hit,
                )
            )
    finally:
        unload_hf_model_bundle(model)

    return BehavioralEvalResult(
        sample_id=sample.sample_id,
        boundary_id=sample.boundary.boundary_id,
        prompt_set=prompt_set,
        keys_per_head=keys_per_head,
        key_selection_method=key_selection_method,
        train_fraction=TRAIN_FRACTION,
        beta_solver=BETA_SOLVER,
        beta_regularization_strength=BETA_REGULARIZATION,
        value_regularization_strength=VALUE_REGULARIZATION,
        prompt_labels=[prompt.label for prompt in prompts],
        reference=_build_path_result(
            path="full_cache_reference",
            keys_per_head=keys_per_head,
            compacted_layers=None,
            prompt_results=reference_runs,
            runtime_seconds=reference_total_runtime,
            prefix_token_count=sample.boundary.prefix_token_count,
        ),
        sketch=_build_path_result(
            path=sketch_selection.source,
            keys_per_head=keys_per_head,
            compacted_layers=sketch_layers,
            prompt_results=sketch_runs,
            runtime_seconds=sketch_total_runtime,
            prefix_token_count=sample.boundary.prefix_token_count,
        ),
        control=_build_path_result(
            path=control_selection.source,
            keys_per_head=keys_per_head,
            compacted_layers=control_layers,
            prompt_results=control_runs,
            runtime_seconds=control_total_runtime,
            prefix_token_count=sample.boundary.prefix_token_count,
        ),
    )
