import csv
import json
import math
import re
import time
from dataclasses import asdict, dataclass, replace
from html import escape
from pathlib import Path
from string import Template
from typing import Any, Literal, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from captum.attr import FeatureAblation, LayerIntegratedGradients, visualization as viz
from transformers import (
    AutoModel,
    AutoModelForQuestionAnswering,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

from src import surprisal


PipelineName = Literal["masking", "attention_viz", "surprisal", "qa", "nli", "decision"]
PipelineResult = dict[str, object]


@dataclass
class PipelineConfig:
    bert_model_name: str = "allegro/herbert-base-cased"
    nli_model_name: str = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"
    qa_model_name: str = "henryk/bert-base-multilingual-cased-finetuned-polish-squad2"
    qa_english_model_name: str = "deepset/roberta-base-squad2"
    surprisal_model: str = "gpt2"
    device: str | None = None
    output_dir: str = "data/interpretability"
    save_outputs: bool = True
    pipelines_to_run: tuple[PipelineName, ...] = ("masking", "attention_viz")
    batch_size: int = 4
    normalize_scores: bool = False
    attention_scope: Literal["summary", "matrices", "both"] = "both"
    attention_viz_max_tokens: int = 48
    attention_viz_max_heads: int = 9
    qa_methods: tuple[Literal["lig", "feature_ablation"], ...] = (
        "lig",
        "feature_ablation",
    )
    qa_target: Literal["start", "end", "both"] = "both"
    hypothesis_template: str = "This text is {label}."
    nli_hypothesis_label: str = "complex"
    viz_output: Literal["html", "none"] = "html"


ALLOWED_PIPELINES: set[str] = {"masking", "attention_viz", "surprisal", "qa", "nli", "decision"}
PIPELINE_ALIASES: dict[str, str] = {"nli": "decision"}
QA_METHODS: set[str] = {"lig", "feature_ablation"}
TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"


class InterpretabilityPipeline:
    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.device = _resolve_device(config.device)
        self._tokenizers: dict[str, Any] = {}
        self._models: dict[tuple[str, str], Any] = {}

    def run(self, text: str, question: str | None = None, context: str | None = None) -> PipelineResult:
        selected = _resolve_pipelines_to_run(self.config)
        if "qa" in selected and (not question or not context):
            raise ValueError("The 'qa' pipeline requires both question and context.")

        run_id = time.strftime("%Y%m%d%H%M%S", time.localtime())
        slug = _slugify(text)
        out_dir = _ensure_output_dir(self.config.output_dir) / f"{run_id}_{slug}"
        if self.config.save_outputs:
            out_dir.mkdir(parents=True, exist_ok=True)

        result: PipelineResult = {}
        if "masking" in selected:
            result["masking"] = self.run_masking(text)
            if self.config.save_outputs:
                _save_masking_outputs(out_dir, result["masking"], self.config)

        if "attention_viz" in selected:
            result["attention_viz"] = self.run_attention_viz(text)
            if self.config.save_outputs:
                _save_attention_outputs(out_dir, result["attention_viz"], self.config)

        if "decision" in selected:
            result["decision"] = self.run_nli_decision(text)
            if self.config.save_outputs:
                _save_decision_outputs(out_dir, result["decision"], self.config)

        if "surprisal" in selected:
            result["surprisal"] = self.run_surprisal(text)
            if self.config.save_outputs:
                _save_surprisal_outputs(out_dir, result["surprisal"])

        if "qa" in selected:
            result["qa"] = self.run_qa(str(question), str(context))
            if self.config.save_outputs:
                _save_qa_outputs(out_dir, result["qa"], self.config)

        if self.config.save_outputs:
            payload: dict[str, object] = {
                "run_id": run_id,
                "text_slug": slug,
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime()),
                "config": asdict(self.config),
                "pipelines_selected": sorted(selected),
                "result": result,
            }
            _save_json(
                out_dir / "pipeline_result.json",
                _serialize_with_tensor_sidecars(payload, out_dir, "pipeline_result"),
            )

        return result

    def run_masking(self, text: str) -> dict[str, object]:
        tokenizer = self._tokenizer(self.config.bert_model_name)
        model = self._encoder_model(self.config.bert_model_name)

        encoded = tokenizer(text, return_tensors="pt", truncation=True)
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = _optional_tensor_to_device(encoded.get("attention_mask"), self.device)
        model_inputs = _model_inputs(input_ids, attention_mask)

        with torch.no_grad():
            original_outputs = model(**model_inputs)
            original_cls = original_outputs.last_hidden_state[:, 0, :]

        tokens = tokenizer.convert_ids_to_tokens(input_ids[0].tolist())
        special_ids = set(getattr(tokenizer, "all_special_ids", []) or [])
        mask_id = tokenizer.mask_token_id
        if mask_id is None:
            raise ValueError(f"Tokenizer for {self.config.bert_model_name} has no mask token.")

        candidate_indices = [
            index
            for index, token_id in enumerate(input_ids[0].tolist())
            if token_id not in special_ids
        ]
        importance = [0.0 for _ in tokens]
        cosine_scores = [1.0 for _ in tokens]

        for batch_indices in _batch_iter(candidate_indices, self.config.batch_size):
            masked_inputs = input_ids.repeat(len(batch_indices), 1)
            for row_index, token_index in enumerate(batch_indices):
                masked_inputs[row_index, token_index] = mask_id

            batched_attention = (
                attention_mask.repeat(len(batch_indices), 1)
                if attention_mask is not None
                else None
            )
            with torch.no_grad():
                outputs = model(**_model_inputs(masked_inputs, batched_attention))
                masked_cls = outputs.last_hidden_state[:, 0, :]
                cosine = F.cosine_similarity(original_cls, masked_cls)

            for row_index, token_index in enumerate(batch_indices):
                score = float(cosine[row_index].item())
                cosine_scores[token_index] = score
                importance[token_index] = 1.0 - score

        if self.config.normalize_scores:
            importance = _normalize_scores(importance)

        return {
            "model_name": self.config.bert_model_name,
            "tokens": tokens,
            "importance": importance,
            "cosine_similarity": cosine_scores,
            "method": "mask_one_token_cls_cosine_shift",
        }

    def run_attention_viz(self, text: str) -> dict[str, object]:
        _validate_attention_scope(self.config.attention_scope)
        tokenizer = self._tokenizer(self.config.bert_model_name)
        model = self._encoder_model(self.config.bert_model_name)

        encoded = tokenizer(text, return_tensors="pt", truncation=True)
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = _optional_tensor_to_device(encoded.get("attention_mask"), self.device)
        model_inputs = _model_inputs(input_ids, attention_mask)
        model_inputs["output_attentions"] = True

        with torch.no_grad():
            outputs = model(**model_inputs)

        raw_attentions = getattr(outputs, "attentions", None)
        if not raw_attentions:
            raise RuntimeError(
                f"Model {self.config.bert_model_name} did not return attention tensors."
            )

        tokens = tokenizer.convert_ids_to_tokens(input_ids[0].tolist())
        stacked = torch.stack([item.detach().cpu() for item in raw_attentions], dim=0)[:, 0]

        scores: list[float] = []
        if self.config.attention_scope in {"summary", "both"}:
            mean_map = stacked.mean(dim=(0, 1))
            incoming = mean_map.mean(dim=0)
            outgoing = mean_map.mean(dim=1)
            merged = (incoming + outgoing) / 2.0
            scores = [float(value.item()) for value in merged]
            if self.config.normalize_scores:
                scores = _normalize_scores(scores)

        matrices = stacked if self.config.attention_scope in {"matrices", "both"} else None
        return {
            "model_name": self.config.bert_model_name,
            "tokens": tokens,
            "attention_token_attributions": scores,
            "attention_matrices": matrices,
            "attention_matrices_path": None,
            "attention_source": "model_output_attentions",
            "attention_aggregation": {
                "matrix": "mean_layers_heads",
                "token_summary": "incoming_outgoing_mean",
            },
            "note": "attention maps are descriptive, not causal attributions",
        }

    def run_nli_decision(self, text: str) -> dict[str, object]:
        tokenizer = self._tokenizer(self.config.nli_model_name)
        model = self._sequence_classifier_model(self.config.nli_model_name)

        hypothesis = self.config.hypothesis_template.format(
            label=self.config.nli_hypothesis_label,
            text=text,
        )
        encoded = tokenizer(text, hypothesis, return_tensors="pt", truncation=True)
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = _optional_tensor_to_device(encoded.get("attention_mask"), self.device)
        token_type_ids = _optional_tensor_to_device(encoded.get("token_type_ids"), self.device)

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)

        target_index = _infer_label_index(model, "entail")
        pred_index = int(torch.argmax(probs, dim=-1).item())
        pred_label = _label_name(model, pred_index)
        label_probs = {str(index): float(prob.item()) for index, prob in enumerate(probs[0])}

        def forward_func(
            forward_input_ids: torch.Tensor,
            forward_attention_mask: torch.Tensor | None = None,
            forward_token_type_ids: torch.Tensor | None = None,
        ) -> torch.Tensor:
            output = model(
                input_ids=forward_input_ids,
                attention_mask=forward_attention_mask,
                token_type_ids=forward_token_type_ids,
            )
            return output.logits[:, target_index]

        baseline_ids = torch.full_like(input_ids, _baseline_token_id(tokenizer))
        lig = LayerIntegratedGradients(forward_func, model.get_input_embeddings())
        attributions, delta = lig.attribute(
            inputs=input_ids,
            baselines=baseline_ids,
            additional_forward_args=(attention_mask, token_type_ids),
            return_convergence_delta=True,
        )
        token_scores = attributions.sum(dim=-1).squeeze(0)
        scores = [float(value.item()) for value in token_scores]
        if self.config.normalize_scores:
            scores = _normalize_scores(scores)

        return {
            "model_name": self.config.nli_model_name,
            "hypothesis": hypothesis,
            "target_label": _label_name(model, target_index),
            "target_label_index": target_index,
            "tokens": tokenizer.convert_ids_to_tokens(input_ids[0].tolist()),
            "attributions": scores,
            "pred_label": pred_label,
            "label_probs": label_probs,
            "lig_delta": float(delta.squeeze().item()),
            "note": (
                "NLI scores explain the entailment logit for the configured hypothesis; "
                "they are not direct readability measurements."
            ),
        }

    def run_surprisal(self, text: str) -> dict[str, object]:
        model, tokenizer, model_name, device = surprisal.load_model_and_tokenizer(
            self.config.surprisal_model,
            device=self.device,
            model_cache=self._models,
            tokenizer_cache=self._tokenizers,
        )
        tokens, surprisals = surprisal.compute_surprisal(text, model, tokenizer, device)
        finite = [float(value) for value in surprisals if math.isfinite(float(value))]
        summary: dict[str, object] = {"token_count": len(tokens)}
        if finite:
            sorted_values = sorted(finite)
            mean_value = sum(sorted_values) / len(sorted_values)
            variance = sum((value - mean_value) ** 2 for value in sorted_values) / len(
                sorted_values
            )
            summary.update(
                {
                    "mean_bits": float(mean_value),
                    "median_bits": float(sorted_values[len(sorted_values) // 2]),
                    "std_bits": float(math.sqrt(variance)),
                    "min_bits": float(sorted_values[0]),
                    "max_bits": float(sorted_values[-1]),
                    "p90_bits": float(
                        sorted_values[min(len(sorted_values) - 1, round(0.9 * (len(sorted_values) - 1)))]
                    ),
                }
            )

        return {
            "model_key": self.config.surprisal_model,
            "model_name": model_name,
            "tokens": tokens,
            "surprisals": [float(value) for value in surprisals],
            "unit": "bits",
            "summary": summary,
        }

    def run_qa(self, question: str, context: str) -> dict[str, object]:
        method_names = {method.strip().lower() for method in self.config.qa_methods}
        unknown_methods = method_names - QA_METHODS
        if unknown_methods:
            raise ValueError(f"Unknown QA attribution method(s): {sorted(unknown_methods)}")

        qa_language = _infer_qa_language(question, context)
        qa_model_name = _resolve_qa_model_name(self.config, qa_language)
        tokenizer = self._tokenizer(qa_model_name)
        model = self._qa_model(qa_model_name)
        pad_id = tokenizer.pad_token_id
        if pad_id is None:
            raise ValueError(f"Tokenizer for {qa_model_name} has no pad token.")

        encoded = tokenizer(question, context, return_tensors="pt", truncation=True)
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = _optional_tensor_to_device(encoded.get("attention_mask"), self.device)
        token_type_ids = _optional_tensor_to_device(encoded.get("token_type_ids"), self.device)

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )
            start_logits = outputs.start_logits[0]
            end_logits = outputs.end_logits[0]

        start_idx = int(torch.argmax(start_logits).item())
        end_idx = start_idx + int(torch.argmax(end_logits[start_idx:]).item())
        start_probs = torch.softmax(start_logits, dim=-1)
        end_probs = torch.softmax(end_logits, dim=-1)
        start_prob = float(start_probs[start_idx].item())
        end_prob = float(end_probs[end_idx].item())
        span_score = start_prob * end_prob

        answer_ids = input_ids[0, start_idx : end_idx + 1]
        predicted_answer = tokenizer.decode(answer_ids, skip_special_tokens=True).strip()
        tokens = tokenizer.convert_ids_to_tokens(input_ids[0].tolist())
        display_tokens = _display_model_tokens(tokens)

        result: dict[str, object] = {
            "model_name": qa_model_name,
            "qa_language": qa_language,
            "question": question,
            "context": context,
            "tokens": tokens,
            "display_tokens": display_tokens,
            "predicted_answer": predicted_answer,
            "start_idx": start_idx,
            "end_idx": end_idx,
            "start_prob": start_prob,
            "end_prob": end_prob,
            "span_score": span_score,
            "qa_target": self.config.qa_target,
        }

        def forward_start(
            forward_input_ids: torch.Tensor,
            forward_attention_mask: torch.Tensor | None = None,
            forward_token_type_ids: torch.Tensor | None = None,
        ) -> torch.Tensor:
            output = model(
                input_ids=forward_input_ids,
                attention_mask=forward_attention_mask,
                token_type_ids=forward_token_type_ids,
            )
            return output.start_logits[:, start_idx]

        def forward_end(
            forward_input_ids: torch.Tensor,
            forward_attention_mask: torch.Tensor | None = None,
            forward_token_type_ids: torch.Tensor | None = None,
        ) -> torch.Tensor:
            output = model(
                input_ids=forward_input_ids,
                attention_mask=forward_attention_mask,
                token_type_ids=forward_token_type_ids,
            )
            return output.end_logits[:, end_idx]

        baseline_ids = torch.full_like(input_ids, int(pad_id))
        forward_args = (attention_mask, token_type_ids)

        if "lig" in method_names:
            lig_start = LayerIntegratedGradients(forward_start, model.get_input_embeddings())
            lig_end = LayerIntegratedGradients(forward_end, model.get_input_embeddings())
            start_attr, start_delta = lig_start.attribute(
                inputs=input_ids,
                baselines=baseline_ids,
                additional_forward_args=forward_args,
                return_convergence_delta=True,
            )
            end_attr, end_delta = lig_end.attribute(
                inputs=input_ids,
                baselines=baseline_ids,
                additional_forward_args=forward_args,
                return_convergence_delta=True,
            )
            start_scores = start_attr.sum(dim=-1).squeeze(0)
            end_scores = end_attr.sum(dim=-1).squeeze(0)
            merged_scores = _aggregate_qa_attributions(
                start_scores,
                end_scores,
                self.config.qa_target,
            )
            lig_scores = [float(value.item()) for value in merged_scores]
            if self.config.normalize_scores:
                lig_scores = _normalize_scores(lig_scores)
            result.update(
                {
                    "lig_start_attributions": [float(value.item()) for value in start_scores],
                    "lig_end_attributions": [float(value.item()) for value in end_scores],
                    "lig_attributions": lig_scores,
                    "lig_delta_start": float(start_delta.squeeze().item()),
                    "lig_delta_end": float(end_delta.squeeze().item()),
                }
            )

        if "feature_ablation" in method_names:
            fa_start = FeatureAblation(forward_start)
            fa_end = FeatureAblation(forward_end)
            start_ablation = fa_start.attribute(
                inputs=input_ids,
                baselines=baseline_ids,
                additional_forward_args=forward_args,
            ).squeeze(0)
            end_ablation = fa_end.attribute(
                inputs=input_ids,
                baselines=baseline_ids,
                additional_forward_args=forward_args,
            ).squeeze(0)
            start_ablation = start_ablation.to(torch.float32)
            end_ablation = end_ablation.to(torch.float32)
            merged_ablation = _aggregate_qa_attributions(
                start_ablation,
                end_ablation,
                self.config.qa_target,
            )
            fa_scores = [float(value.item()) for value in merged_ablation]
            if self.config.normalize_scores:
                fa_scores = _normalize_scores(fa_scores)
            result.update(
                {
                    "feature_ablation_start_attributions": [
                        float(value.item()) for value in start_ablation
                    ],
                    "feature_ablation_end_attributions": [
                        float(value.item()) for value in end_ablation
                    ],
                    "feature_ablation_attributions": fa_scores,
                }
            )

        return result

    def _tokenizer(self, model_name: str) -> Any:
        if model_name not in self._tokenizers:
            self._tokenizers[model_name] = AutoTokenizer.from_pretrained(model_name)
        return self._tokenizers[model_name]

    def _encoder_model(self, model_name: str) -> Any:
        key = ("encoder", model_name)
        if key not in self._models:
            self._models[key] = AutoModel.from_pretrained(
                model_name,
                attn_implementation="eager",
            )
        model = self._models[key].to(self.device)
        model.eval()
        return model

    def _qa_model(self, model_name: str) -> Any:
        key = ("qa", model_name)
        if key not in self._models:
            self._models[key] = AutoModelForQuestionAnswering.from_pretrained(
                model_name,
                attn_implementation="eager",
            )
        model = self._models[key].to(self.device)
        model.eval()
        model.zero_grad()
        return model

    def _sequence_classifier_model(self, model_name: str) -> Any:
        key = ("sequence_classification", model_name)
        if key not in self._models:
            self._models[key] = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                attn_implementation="eager",
            )
        model = self._models[key].to(self.device)
        model.eval()
        model.zero_grad()
        return model


def run_pipeline(
    text: str,
    config: PipelineConfig,
    question: str | None = None,
    context: str | None = None,
) -> PipelineResult:
    return InterpretabilityPipeline(config).run(text, question=question, context=context)


def run_batch(
    texts: Sequence[str],
    config: PipelineConfig,
    question: str | None = None,
    context: str | None = None,
) -> list[PipelineResult]:
    batch_run_id = time.strftime("%Y%m%d%H%M%S", time.localtime())
    batch_dir = _ensure_output_dir(config.output_dir) / f"batch_{batch_run_id}"
    if config.save_outputs:
        batch_dir.mkdir(parents=True, exist_ok=True)

    results: list[PipelineResult] = []
    manifest: dict[str, object] = {
        "batch_run_id": batch_run_id,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime()),
        "output_dir": str(batch_dir),
        "items": [],
    }
    for index, text in enumerate(texts):
        item_dir = batch_dir / f"{index:04d}_{_slugify(text)}"
        item_config = replace(config, output_dir=str(item_dir))
        result = run_pipeline(text, item_config, question=question, context=context)
        results.append(result)
        manifest["items"].append(
            {
                "index": index,
                "text_slug": _slugify(text),
                "run_dir": str(item_dir),
                "result_keys": sorted(result.keys()),
            }
        )

    if config.save_outputs:
        _save_json(batch_dir / "batch_manifest.json", manifest)
    return results


def load_texts_from_dir(path: str, glob: str = "**/*.txt") -> list[str]:
    root = Path(path)
    return [
        file.read_text(encoding="utf-8").replace("\n", " ").strip()
        for file in root.glob(glob)
        if file.is_file()
    ]


def run_gaze_alignment_benchmark(
    gaze_token_stats_path: str | Path,
    pipeline_result_path: str | Path,
    output_dir: str | Path,
    item_id: str | None = None,
    participant_id: str | None = None,
    frequency_csv: str | Path | None = None,
) -> dict[str, object]:
    from src.gaze_model_alignment_benchmark import (
        BenchmarkPair,
        run_benchmark as run_gaze_benchmark,
    )

    benchmark_result = run_gaze_benchmark(
        pairs=[
            BenchmarkPair(
                gaze_token_stats_path=Path(gaze_token_stats_path),
                pipeline_result_path=Path(pipeline_result_path),
                item_id=item_id,
                participant_id=participant_id,
            )
        ],
        output_dir=Path(output_dir),
        frequency_csv=Path(frequency_csv) if frequency_csv is not None else None,
    )
    return benchmark_result.report


def _resolve_pipelines_to_run(config: PipelineConfig) -> set[str]:
    requested = {
        PIPELINE_ALIASES.get(str(name).strip().lower(), str(name).strip().lower())
        for name in config.pipelines_to_run
    }
    if not requested:
        raise ValueError("pipelines_to_run must contain at least one pipeline name.")
    unknown = requested - ALLOWED_PIPELINES
    if unknown:
        raise ValueError(
            f"Unknown pipeline stage(s): {sorted(unknown)}. "
            f"Allowed: {sorted(ALLOWED_PIPELINES)}"
        )
    return set(requested)


def _resolve_device(device: str | None) -> torch.device:
    if device:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _infer_qa_language(question: str, context: str) -> Literal["en", "pl"]:
    text = f"{question} {context}".lower()
    if re.search(r"[ąćęłńóśźż]", text):
        return "pl"

    polish_markers = {
        "czy",
        "czym",
        "jak",
        "jakie",
        "jaki",
        "jakim",
        "jest",
        "ktory",
        "ktore",
        "ktorych",
        "nie",
        "oraz",
        "sie",
        "w",
        "z",
        "ze",
    }
    tokens = set(re.findall(r"\b\w+\b", text, flags=re.UNICODE))
    if len(tokens & polish_markers) >= 2:
        return "pl"
    return "en"


def _resolve_qa_model_name(config: PipelineConfig, language: Literal["en", "pl"]) -> str:
    if language == "en":
        return config.qa_english_model_name
    return config.qa_model_name


def _model_inputs(input_ids: torch.Tensor, attention_mask: torch.Tensor | None) -> dict[str, Any]:
    values = {"input_ids": input_ids}
    if attention_mask is not None:
        values["attention_mask"] = attention_mask
    return values


def _optional_tensor_to_device(value: Any, device: torch.device) -> torch.Tensor | None:
    if value is None:
        return None
    return value.to(device)


def _infer_label_index(model: Any, label_fragment: str) -> int:
    fragment = label_fragment.lower()
    label2id = getattr(model.config, "label2id", {}) or {}
    if isinstance(label2id, dict):
        for label, index in label2id.items():
            if fragment in str(label).lower():
                return int(index)

    id2label = getattr(model.config, "id2label", {}) or {}
    if isinstance(id2label, dict):
        for index, label in id2label.items():
            if fragment in str(label).lower():
                return int(index)

    return 0


def _label_name(model: Any, label_index: int) -> str:
    id2label = getattr(model.config, "id2label", {}) or {}
    if isinstance(id2label, dict):
        return str(id2label.get(label_index, id2label.get(str(label_index), label_index)))
    return str(label_index)


def _baseline_token_id(tokenizer: Any) -> int:
    for attr_name in ("pad_token_id", "eos_token_id", "sep_token_id", "cls_token_id"):
        token_id = getattr(tokenizer, attr_name, None)
        if token_id is not None:
            return int(token_id)
    return 0


def _batch_iter(indices: Sequence[int], batch_size: int) -> list[list[int]]:
    safe_batch_size = max(1, int(batch_size))
    return [
        list(indices[start : start + safe_batch_size])
        for start in range(0, len(indices), safe_batch_size)
    ]


def _normalize_scores(scores: Sequence[float]) -> list[float]:
    denom = sum(abs(score) for score in scores)
    if denom == 0.0:
        return [float(score) for score in scores]
    return [float(score) / denom for score in scores]


def _validate_attention_scope(scope: str) -> None:
    if scope not in {"summary", "matrices", "both"}:
        raise ValueError("attention_scope must be 'summary', 'matrices', or 'both'.")


def _aggregate_qa_attributions(start_attr: torch.Tensor, end_attr: torch.Tensor, target_mode: str) -> torch.Tensor:
    if target_mode == "start":
        return start_attr
    if target_mode == "end":
        return end_attr
    if target_mode == "both":
        return (start_attr + end_attr) / 2.0
    raise ValueError("qa_target must be 'start', 'end', or 'both'.")


def _display_model_tokens(tokens: Sequence[object]) -> list[str]:
    return [_display_model_token(str(token)) for token in tokens]


def _display_model_token(token: str) -> str:
    text = token
    for marker in ("\u00e2\u2013\u0081", "\u0120", "\u2581"):
        if text.startswith(marker):
            text = text[len(marker) :]
    if text.startswith("##"):
        text = text[2:]
    return _repair_token_mojibake(text)


def _repair_token_mojibake(token: str) -> str:
    candidates = [token]
    for encoding in ("cp1250", "cp1252", "latin1"):
        try:
            candidates.append(token.encode(encoding).decode("utf-8"))
        except UnicodeError:
            continue
    return min(candidates, key=_token_mojibake_score)


def _token_mojibake_score(token: str) -> int:
    markers = ("Â", "Ã", "Ä", "Å", "Ĺ", "â", "�")
    return sum(token.count(marker) for marker in markers)


def _slugify(text: str, max_len: int = 60) -> str:
    slug = re.sub(r"\s+", "_", text.strip().lower())
    slug = re.sub(r"[^a-z0-9_\-]+", "", slug)
    return (slug[:max_len].rstrip("_") or "sample")


def _ensure_output_dir(path: str | Path) -> Path:
    out_dir = Path(path)
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _save_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _save_csv(path: Path, tokens: Sequence[str], scores: Sequence[float]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["token", "score"])
        for token, score in zip(tokens, scores):
            writer.writerow([str(token).replace("\n", " "), float(score)])


def _save_plot(path: Path, tokens: Sequence[str], scores: Sequence[float], title: str) -> None:
    if not tokens:
        return
    plt.figure(figsize=(10, max(3, min(12, len(tokens) * 0.3))))
    y_pos = list(range(len(tokens)))
    plt.barh(y_pos, [float(score) for score in scores])
    plt.yticks(y_pos, [str(token) for token in tokens], fontsize=8)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def _save_masking_outputs(out_dir: Path, masking_result: dict[str, object], config: PipelineConfig) -> None:
    payload = dict(masking_result)
    tokens = [str(token) for token in payload["tokens"]]
    scores = [float(score) for score in payload["importance"]]
    _save_csv(out_dir / "masking.csv", tokens, scores)
    _save_plot(out_dir / "masking.png", tokens, scores, "BERT Masking Importance")
    if config.viz_output == "html":
        html_path = out_dir / "masking.html"
        _save_captum_text_html(
            html_path,
            tokens=tokens,
            scores=scores,
            predicted_label="CLS shift",
            true_label="CLS shift",
            attribution_label="BERT Masking Importance",
            score=float(sum(scores)),
        )
        payload["viz_html_paths"] = [str(html_path)]
    _save_json(out_dir / "masking.json", payload)
    masking_result.update(payload)


def _save_attention_outputs(out_dir: Path, attention_result: dict[str, object], config: PipelineConfig) -> None:
    payload = dict(attention_result)
    tokens = [str(token) for token in payload.get("tokens", [])]
    scores = [float(score) for score in payload.get("attention_token_attributions", [])]
    matrices = payload.get("attention_matrices")
    viz_dir = out_dir / "attention_viz"
    viz_dir.mkdir(parents=True, exist_ok=True)

    paths: dict[str, object] = {}
    if isinstance(matrices, torch.Tensor):
        matrices_path = out_dir / "attention_viz_matrices.pt"
        torch.save(matrices, matrices_path)
        payload["attention_matrices_path"] = str(matrices_path)
        paths["attention_matrices_pt"] = str(matrices_path)
        _save_attention_matrix_plots(viz_dir, matrices, tokens, config, paths)

    if config.viz_output == "html" and tokens and scores:
        html_path = viz_dir / "attention_tokens.html"
        _save_captum_text_html(
            html_path,
            tokens=tokens,
            scores=scores,
            predicted_label="attention",
            true_label="attention",
            attribution_label="Attention Token Weights",
            score=float(sum(scores)),
        )
        paths["attention_tokens_html"] = str(html_path)

    payload["paths"] = paths
    payload["artifact_dir"] = str(viz_dir)
    json_payload = dict(payload)
    json_payload.pop("attention_matrices", None)
    _save_json(out_dir / "attention_viz.json", json_payload)
    attention_result.update(payload)


def _save_decision_outputs(out_dir: Path, decision_result: dict[str, object], config: PipelineConfig) -> None:
    payload = dict(decision_result)
    tokens = [str(token) for token in payload.get("tokens", [])]
    scores = [float(score) for score in payload.get("attributions", [])]
    _save_csv(out_dir / "decision.csv", tokens, scores)
    _save_plot(out_dir / "decision_lig.png", tokens, scores, "NLI Layer Integrated Gradients")
    if config.viz_output == "html":
        html_path = out_dir / "decision_attribution.html"
        label_probs_payload = payload.get("label_probs", {})
        target_probability = 0.0
        if isinstance(label_probs_payload, dict):
            target_key = str(payload.get("target_label_index", 0))
            target_probability = float(label_probs_payload.get(target_key, 0.0))
        _save_captum_text_html(
            html_path,
            tokens=tokens,
            scores=scores,
            predicted_label=str(payload.get("pred_label", "")),
            true_label=str(payload.get("target_label", "")),
            attribution_label="NLI Entailment LIG",
            score=target_probability,
        )
        payload["viz_html_paths"] = [str(html_path)]
    _save_json(out_dir / "decision.json", payload)
    decision_result.update(payload)


def _save_attention_matrix_plots(
    viz_dir: Path,
    matrices: torch.Tensor,
    tokens: Sequence[str],
    config: PipelineConfig,
    paths: dict[str, object],
) -> None:
    matrix_np = matrices.detach().cpu().float().numpy()
    max_tokens = min(max(4, int(config.attention_viz_max_tokens)), len(tokens))
    max_tokens = min(max_tokens, matrix_np.shape[-1], matrix_np.shape[-2])
    token_labels = [str(token).replace("\n", " ")[:24] for token in tokens[:max_tokens]]

    if matrix_np.ndim == 4:
        head_count = min(matrix_np.shape[1], max(1, int(config.attention_viz_max_heads)))
        columns = min(3, head_count)
        rows = int(math.ceil(head_count / columns))
        figure = plt.figure(figsize=(6 * columns, 5 * rows))
        for head_index in range(head_count):
            axis = figure.add_subplot(rows, columns, head_index + 1)
            head_matrix = matrix_np[0, head_index, :max_tokens, :max_tokens]
            image = axis.imshow(head_matrix, cmap="viridis", aspect="auto")
            axis.set_title(f"Head {head_index + 1}")
            axis.set_xticks(range(max_tokens))
            axis.set_yticks(range(max_tokens))
            axis.set_xticklabels(token_labels, fontsize=7, rotation=90)
            axis.set_yticklabels(token_labels, fontsize=7)
            figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
        figure.tight_layout()
        heads_path = viz_dir / "token2token_heads.png"
        figure.savefig(heads_path, dpi=220)
        plt.close(figure)
        paths["token2token_heads_png"] = str(heads_path)
        flow_matrix = matrix_np.mean(axis=(0, 1))
    elif matrix_np.ndim == 3:
        flow_matrix = matrix_np.mean(axis=0)
    else:
        flow_matrix = matrix_np

    cropped = flow_matrix[:max_tokens, :max_tokens]
    figure, axis = plt.subplots(figsize=(max(6, max_tokens * 0.35), max(5, max_tokens * 0.3)))
    image = axis.imshow(cropped, cmap="viridis", aspect="auto")
    axis.set_title("Mean Attention Flow")
    axis.set_xticks(range(max_tokens))
    axis.set_yticks(range(max_tokens))
    axis.set_xticklabels(token_labels, fontsize=7, rotation=90)
    axis.set_yticklabels(token_labels, fontsize=7)
    axis.set_xlabel("Token attended to")
    axis.set_ylabel("Token attending")
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.tight_layout()
    flow_path = viz_dir / "attention_flow.png"
    figure.savefig(flow_path, dpi=240)
    plt.close(figure)
    paths["attention_flow_png"] = str(flow_path)


def _save_qa_outputs(out_dir: Path, qa_result: dict[str, object], config: PipelineConfig) -> None:
    payload = dict(qa_result)
    raw_tokens = [str(token) for token in payload.get("tokens", [])]
    display_tokens = [str(token) for token in payload.get("display_tokens", [])]
    tokens = display_tokens if len(display_tokens) == len(raw_tokens) else _display_model_tokens(raw_tokens)
    score_specs = [
        ("lig_attributions", "qa_lig.csv", "qa_lig.png", "QA Layer Integrated Gradients"),
        (
            "feature_ablation_attributions",
            "qa_feature_ablation.csv",
            "qa_feature_ablation.png",
            "QA Feature Ablation",
        ),
    ]
    for score_key, csv_name, png_name, title in score_specs:
        scores = payload.get(score_key)
        if isinstance(scores, list):
            numeric_scores = [float(score) for score in scores]
            _save_csv(out_dir / csv_name, tokens, numeric_scores)
            _save_plot(out_dir / png_name, tokens, numeric_scores, title)

    if config.viz_output == "html":
        html_paths: list[str] = []
        for score_key, filename, label in [
            ("lig_start_attributions", "qa_lig_start.html", "QA LIG Start"),
            ("lig_end_attributions", "qa_lig_end.html", "QA LIG End"),
            ("lig_attributions", "qa_lig.html", "QA LIG Aggregated"),
            (
                "feature_ablation_start_attributions",
                "qa_feature_ablation_start.html",
                "QA Feature Ablation Start",
            ),
            (
                "feature_ablation_end_attributions",
                "qa_feature_ablation_end.html",
                "QA Feature Ablation End",
            ),
            (
                "feature_ablation_attributions",
                "qa_feature_ablation.html",
                "QA Feature Ablation Aggregated",
            ),
        ]:
            scores = payload.get(score_key)
            if isinstance(scores, list):
                html_path = out_dir / filename
                _save_captum_text_html(
                    html_path,
                    tokens=tokens,
                    scores=[float(score) for score in scores],
                    predicted_label=str(payload.get("predicted_answer", "")),
                    true_label=str(payload.get("predicted_answer", "")),
                    attribution_label=label,
                    score=float(payload.get("span_score", 0.0)),
                )
                html_paths.append(str(html_path))
        if html_paths:
            payload["viz_html_paths"] = html_paths
    _save_json(out_dir / "qa.json", payload)
    qa_result.update(payload)


def _save_captum_text_html(
    path: Path,
    tokens: Sequence[str],
    scores: Sequence[float],
    predicted_label: str,
    true_label: str,
    attribution_label: str,
    score: float,
) -> None:
    score_tensor = torch.tensor([float(value) for value in scores], dtype=torch.float32)
    record = viz.VisualizationDataRecord(
        score_tensor,
        score,
        predicted_label,
        true_label,
        attribution_label,
        float(score_tensor.sum().item()),
        list(tokens),
        None,
    )
    html_object = viz.visualize_text([record])
    html = html_object.data if hasattr(html_object, "data") else str(html_object)
    path.write_text(html, encoding="utf-8")


def _save_surprisal_outputs(out_dir: Path, result: dict[str, object]) -> None:
    payload = dict(result)
    _save_json(out_dir / "surprisal.json", payload)
    _save_surprisal_csv(out_dir / "surprisal.csv", payload)
    text_path = out_dir / "surprisal.txt"
    _save_surprisal_text_report(text_path, payload)
    html_path = out_dir / "surprisal.html"
    _save_surprisal_html(html_path, payload)
    plot_paths = _save_surprisal_plots(out_dir, payload)
    payload["artifacts"] = {
        "text_report_path": str(text_path),
        "html_report_path": str(html_path),
        "plot_paths": plot_paths,
    }
    result.update(payload)


def _save_surprisal_csv(path: Path, surprisal_result: dict[str, object]) -> None:
    tokens = list(surprisal_result.get("tokens", []))
    surprisals = list(surprisal_result.get("surprisals", []))
    items = [{"item": 1, "surprisals": list(zip(tokens, surprisals))}]
    with path.open("w", encoding="utf-8") as handle:
        surprisal.write_surprisal_csv(items, handle)


def _save_surprisal_text_report(path: Path, surprisal_result: dict[str, object]) -> None:
    tokens = list(surprisal_result.get("tokens", []))
    surprisals = list(surprisal_result.get("surprisals", []))
    items = [{"item": 1, "surprisals": list(zip(tokens, surprisals))}]
    with path.open("w", encoding="utf-8") as handle:
        surprisal.pretty_print_surprisals(items, handle)


def _save_surprisal_plots(out_dir: Path, surprisal_result: dict[str, object]) -> list[str]:
    tokens = list(surprisal_result.get("tokens", []))
    surprisals = list(surprisal_result.get("surprisals", []))
    pairs = [
        (index, str(token).replace("\n", " "), float(value))
        for index, (token, value) in enumerate(zip(tokens, surprisals), start=1)
        if math.isfinite(float(value))
    ]
    if not pairs:
        return []

    profile_path = out_dir / "surprisal_profile.png"
    plt.figure(figsize=(12, 4))
    plt.plot([item[0] for item in pairs], [item[2] for item in pairs], marker="o")
    plt.xlabel("Token Index")
    plt.ylabel("Surprisal (bits)")
    plt.title("Token Surprisal Profile")
    plt.grid(alpha=0.25, linestyle="--")
    plt.tight_layout()
    plt.savefig(profile_path, dpi=200)
    plt.close()

    top_pairs = list(reversed(sorted(pairs, key=lambda item: item[2], reverse=True)[:20]))
    top_path = out_dir / "surprisal_top_tokens.png"
    plt.figure(figsize=(10, max(4, len(top_pairs) * 0.35)))
    plt.barh(range(len(top_pairs)), [item[2] for item in top_pairs])
    plt.yticks(
        range(len(top_pairs)),
        [f"{index}:{token[:30]}" for index, token, _ in top_pairs],
        fontsize=8,
    )
    plt.xlabel("Surprisal (bits)")
    plt.title("Top High-Surprisal Tokens")
    plt.tight_layout()
    plt.savefig(top_path, dpi=200)
    plt.close()
    return [str(profile_path), str(top_path)]


def _save_surprisal_html(path: Path, surprisal_result: dict[str, object]) -> str:
    template = Template((TEMPLATE_DIR / "surprisal_report.html").read_text(encoding="utf-8"))
    tokens = list(surprisal_result.get("tokens", []))
    surprisals = list(surprisal_result.get("surprisals", []))
    summary = dict(surprisal_result.get("summary", {}))
    pairs = [
        (index, str(token), float(value))
        for index, (token, value) in enumerate(zip(tokens, surprisals), start=1)
        if math.isfinite(float(value))
    ]
    values = [item[2] for item in pairs]
    min_value = min(values) if values else 0.0
    max_value = max(values) if values else 1.0
    token_spans = "\n".join(
        (
            f'<span class="tok" style="background:{_surprisal_bg_color(value, min_value, max_value)}" '
            f'title="idx={index}, surprisal={value:.3f}">{escape(token.replace(chr(10), " "))}</span>'
        )
        for index, token, value in pairs
    )
    top_rows = "\n".join(
        (
            "<tr>"
            f"<td>{index}</td>"
            f"<td>{escape(token.replace(chr(10), ' '))}</td>"
            f"<td>{value:.3f}</td>"
            "</tr>"
        )
        for index, token, value in sorted(pairs, key=lambda item: item[2], reverse=True)[:25]
    )
    html = template.safe_substitute(
        model_name=escape(str(surprisal_result.get("model_name", ""))),
        model_key=escape(str(surprisal_result.get("model_key", ""))),
        unit=escape(str(surprisal_result.get("unit", "bits"))),
        token_count=escape(str(summary.get("token_count", len(tokens)))),
        mean_bits=_format_optional_float(summary.get("mean_bits")),
        median_bits=_format_optional_float(summary.get("median_bits")),
        std_bits=_format_optional_float(summary.get("std_bits")),
        p90_bits=_format_optional_float(summary.get("p90_bits")),
        max_bits=_format_optional_float(summary.get("max_bits")),
        token_spans=token_spans,
        top_rows=top_rows,
    )
    path.write_text(html, encoding="utf-8")
    return str(path)


def _format_optional_float(value: object) -> str:
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return f"{float(value):.3f}"
    return "n/a"


def _surprisal_bg_color(value: float, min_value: float, max_value: float) -> str:
    ratio = 0.5 if max_value <= min_value else (value - min_value) / (max_value - min_value)
    ratio = max(0.0, min(1.0, ratio))
    red = round(70 + (220 - 70) * ratio)
    green = round(140 + (70 - 140) * ratio)
    blue = round(230 + (70 - 230) * ratio)
    return f"rgb({red}, {green}, {blue})"


def _serialize_with_tensor_sidecars(payload: object, out_dir: Path, stem: str) -> object:
    sidecar_dir = out_dir / f"{stem}_sidecars"
    sidecar_dir.mkdir(parents=True, exist_ok=True)
    counter = {"value": 0}

    def visit(value: object, path_key: str) -> object:
        if isinstance(value, torch.Tensor):
            counter["value"] += 1
            relative_path = Path(f"{stem}_sidecars/{path_key}_{counter['value']}.pt")
            torch.save(value.detach().cpu(), out_dir / relative_path)
            return {
                "__type__": "torch.Tensor",
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "sidecar_path": str(relative_path).replace("\\", "/"),
            }
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, dict):
            return {str(key): visit(item, f"{path_key}_{key}") for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [visit(item, f"{path_key}_{index}") for index, item in enumerate(value)]
        if isinstance(value, Path):
            return str(value)
        return value

    return visit(payload, stem)
