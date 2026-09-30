import json
import math
import re
import string
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from src.attn_gaze_correlation import DEFAULT_GAZE_METRICS, normalize_token_for_alignment


FEATURE_ABLATION_COLUMNS: tuple[str, ...] = (
    "qa_feature_ablation",
    "qa_feature_ablation_start",
    "qa_feature_ablation_end",
)


@dataclass(frozen=True)
class QaSourceItem:
    item_id: str
    language: str
    context: str
    question: str
    answer: str
    canonical_key: str


@dataclass(frozen=True)
class AnswerScores:
    exact_match: bool
    token_f1: float
    precision: float
    recall: float
    normalized_gold_answer: str
    normalized_predicted_answer: str


def run_answer_gaze_relationship_analysis(
    qa_source_path: Path,
    qa_gaze_benchmark_dir: Path,
    output_dir: Path,
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    source_by_key = load_qa_source_items(qa_source_path)
    manifest = pd.read_csv(qa_gaze_benchmark_dir / "run_manifest.csv")

    correctness_rows: list[dict[str, object]] = []
    correlation_rows: list[dict[str, object]] = []
    span_rows: list[dict[str, object]] = []

    for _, manifest_row in manifest.iterrows():
        if str(manifest_row.get("status", "")) != "ok":
            continue

        text_id = str(manifest_row["text_id"])
        qa_result_path = Path(str(manifest_row["qa_result_path"]))
        qa_payload = load_json_object(qa_result_path)
        qa_key = qa_display_canonical_key(qa_payload)
        source_item = source_by_key.get(qa_key)
        if source_item is None:
            correctness_rows.append(
                {
                    "text_id": text_id,
                    "qa_result_path": str(qa_result_path),
                    "status": "missing_source_item",
                }
            )
            continue

        aligned_path = qa_gaze_benchmark_dir / text_id / "aligned_tokens.csv"
        aligned = pd.read_csv(aligned_path)
        predicted_answer = str(qa_payload.get("predicted_answer", ""))
        answer_scores = score_answer(source_item.answer, predicted_answer)
        correctness_rows.append(
            build_correctness_row(
                text_id=text_id,
                source_item=source_item,
                qa_payload=qa_payload,
                qa_result_path=qa_result_path,
                aligned_path=aligned_path,
                answer_scores=answer_scores,
            )
        )
        correlation_rows.extend(build_feature_gaze_correlation_rows(text_id, aligned))
        span_rows.extend(
            build_answer_span_rows(
                text_id=text_id,
                source_item=source_item,
                predicted_answer=predicted_answer,
                aligned=aligned,
            )
        )

    correctness = pd.DataFrame(correctness_rows)
    correlations = pd.DataFrame(correlation_rows)
    spans = pd.DataFrame(span_rows)
    summary = build_relationship_summary(correctness, correlations, spans)

    correctness_path = output_dir / "answer_correctness.csv"
    correlations_path = output_dir / "feature_ablation_gaze_correlations.csv"
    spans_path = output_dir / "answer_span_summaries.csv"
    summary_path = output_dir / "relationship_summary.csv"
    report_path = output_dir / "answer_relationship_report.json"

    correctness.to_csv(correctness_path, index=False)
    correlations.to_csv(correlations_path, index=False)
    spans.to_csv(spans_path, index=False)
    summary.to_csv(summary_path, index=False)
    write_report(report_path, correctness, correlations, spans, summary)

    return {
        "answer_correctness_csv": str(correctness_path),
        "feature_ablation_gaze_correlations_csv": str(correlations_path),
        "answer_span_summaries_csv": str(spans_path),
        "relationship_summary_csv": str(summary_path),
        "answer_relationship_report_json": str(report_path),
    }


def load_qa_source_items(path: Path) -> dict[str, QaSourceItem]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"QA source is not a list: {path}")

    items: dict[str, QaSourceItem] = {}
    for raw_item in payload:
        if not isinstance(raw_item, dict):
            continue
        context = repair_mojibake(str(raw_item.get("passage", "")).strip())
        question = repair_mojibake(str(raw_item.get("question", "")).strip())
        answer = repair_mojibake(str(raw_item.get("answer", "")).strip())
        if not context or not question:
            continue
        key = canonical_text_key(format_display_text(context, question))
        items[key] = QaSourceItem(
            item_id=str(raw_item.get("id", "")),
            language=str(raw_item.get("lang", "")).strip().lower(),
            context=context,
            question=question,
            answer=answer,
            canonical_key=key,
        )
    return items


def build_correctness_row(
    text_id: str,
    source_item: QaSourceItem,
    qa_payload: dict[str, Any],
    qa_result_path: Path,
    aligned_path: Path,
    answer_scores: AnswerScores,
) -> dict[str, object]:
    return {
        "status": "ok",
        "text_id": text_id,
        "qa_source_id": source_item.item_id,
        "language": source_item.language,
        "question": source_item.question,
        "gold_answer": source_item.answer,
        "predicted_answer": str(qa_payload.get("predicted_answer", "")),
        "answer_exact_match": answer_scores.exact_match,
        "answer_token_f1": answer_scores.token_f1,
        "answer_precision": answer_scores.precision,
        "answer_recall": answer_scores.recall,
        "normalized_gold_answer": answer_scores.normalized_gold_answer,
        "normalized_predicted_answer": answer_scores.normalized_predicted_answer,
        "start_prob": safe_float(qa_payload.get("start_prob")),
        "end_prob": safe_float(qa_payload.get("end_prob")),
        "span_score": safe_float(qa_payload.get("span_score")),
        "qa_result_path": str(qa_result_path),
        "aligned_tokens_path": str(aligned_path),
    }


def build_feature_gaze_correlation_rows(
    text_id: str,
    aligned: pd.DataFrame,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for feature_column in FEATURE_ABLATION_COLUMNS:
        if feature_column not in aligned.columns:
            continue
        for gaze_metric in DEFAULT_GAZE_METRICS:
            if gaze_metric not in aligned.columns:
                continue
            pair = aligned[[feature_column, gaze_metric]].apply(pd.to_numeric, errors="coerce")
            pair = pair.replace([np.inf, -np.inf], np.nan).dropna()
            bundle = correlation_bundle(pair[feature_column], pair[gaze_metric])
            rows.append(
                {
                    "text_id": text_id,
                    "feature_ablation_metric": feature_column,
                    "gaze_metric": gaze_metric,
                    **bundle,
                }
            )
    return rows


def build_answer_span_rows(
    text_id: str,
    source_item: QaSourceItem,
    predicted_answer: str,
    aligned: pd.DataFrame,
) -> list[dict[str, object]]:
    return [
        build_span_row(
            text_id=text_id,
            span_type="gold",
            answer_text=source_item.answer,
            aligned=aligned,
        ),
        build_span_row(
            text_id=text_id,
            span_type="predicted",
            answer_text=predicted_answer,
            aligned=aligned,
        ),
    ]


def build_span_row(
    text_id: str,
    span_type: str,
    answer_text: str,
    aligned: pd.DataFrame,
) -> dict[str, object]:
    span = locate_answer_span(answer_text, aligned["gaze_token"].astype(str).tolist())
    row: dict[str, object] = {
        "text_id": text_id,
        "span_type": span_type,
        "answer_text": answer_text,
        "span_found": span is not None,
        "span_start": "",
        "span_end": "",
        "span_token_count": 0,
    }
    if span is None:
        return row

    start, end = span
    span_frame = aligned.iloc[start : end + 1].copy()
    row.update(
        {
            "span_start": int(start),
            "span_end": int(end),
            "span_token_count": int(len(span_frame)),
            "span_tokens": " ".join(str(token) for token in span_frame["gaze_token"]),
        }
    )
    for feature_column in FEATURE_ABLATION_COLUMNS:
        if feature_column in span_frame.columns:
            values = pd.to_numeric(span_frame[feature_column], errors="coerce").dropna()
            row[f"{feature_column}_signed_sum"] = float(values.sum())
            row[f"{feature_column}_abs_sum"] = float(values.abs().sum())
            row[f"{feature_column}_mean"] = float(values.mean()) if len(values) else math.nan
    for gaze_metric in DEFAULT_GAZE_METRICS:
        if gaze_metric in span_frame.columns:
            values = pd.to_numeric(span_frame[gaze_metric], errors="coerce").dropna()
            row[f"{gaze_metric}_mean"] = float(values.mean()) if len(values) else math.nan
            row[f"{gaze_metric}_sum"] = float(values.sum()) if len(values) else math.nan
    return row


def build_relationship_summary(
    correctness: pd.DataFrame,
    correlations: pd.DataFrame,
    spans: pd.DataFrame,
) -> pd.DataFrame:
    if correctness.empty:
        return pd.DataFrame()

    rows: list[dict[str, object]] = []
    for _, row in correctness.iterrows():
        text_id = str(row["text_id"])
        text_correlations = correlations[
            (correlations["text_id"] == text_id)
            & (correlations["status"] == "ok")
            & (correlations["feature_ablation_metric"] == "qa_feature_ablation")
        ].copy()
        best = best_relationship_row(text_correlations)
        gold_span = span_lookup(spans, text_id, "gold")
        predicted_span = span_lookup(spans, text_id, "predicted")
        rows.append(
            {
                "text_id": text_id,
                "qa_source_id": row.get("qa_source_id", ""),
                "language": row.get("language", ""),
                "gold_answer": row.get("gold_answer", ""),
                "predicted_answer": row.get("predicted_answer", ""),
                "answer_exact_match": row.get("answer_exact_match", False),
                "answer_token_f1": row.get("answer_token_f1", math.nan),
                "span_score": row.get("span_score", math.nan),
                "strongest_gaze_metric": best.get("gaze_metric", ""),
                "strongest_pearson_r": best.get("pearson_r", math.nan),
                "strongest_spearman_rho": best.get("spearman_rho", math.nan),
                "strongest_relationship_n": best.get("n", 0),
                "gold_span_found": gold_span.get("span_found", False),
                "predicted_span_found": predicted_span.get("span_found", False),
                "gold_qa_feature_ablation_abs_sum": gold_span.get(
                    "qa_feature_ablation_abs_sum", math.nan
                ),
                "predicted_qa_feature_ablation_abs_sum": predicted_span.get(
                    "qa_feature_ablation_abs_sum", math.nan
                ),
                "gold_dwell_time_mean": gold_span.get("dwell_time_mean_mean", math.nan),
                "predicted_dwell_time_mean": predicted_span.get("dwell_time_mean_mean", math.nan),
                "gold_fixation_event_count_mean": gold_span.get(
                    "fixation_event_count_mean_mean", math.nan
                ),
                "predicted_fixation_event_count_mean": predicted_span.get(
                    "fixation_event_count_mean_mean", math.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def best_relationship_row(correlations: pd.DataFrame) -> dict[str, object]:
    if correlations.empty:
        return {}
    ranked = correlations.copy()
    ranked["absolute_spearman"] = pd.to_numeric(ranked["spearman_rho"], errors="coerce").abs()
    ranked["absolute_pearson"] = pd.to_numeric(ranked["pearson_r"], errors="coerce").abs()
    ranked = ranked.sort_values(["absolute_spearman", "absolute_pearson"], ascending=False)
    return ranked.iloc[0].to_dict()


def span_lookup(spans: pd.DataFrame, text_id: str, span_type: str) -> dict[str, object]:
    if spans.empty:
        return {}
    match = spans[(spans["text_id"] == text_id) & (spans["span_type"] == span_type)]
    if match.empty:
        return {}
    return match.iloc[0].to_dict()


def locate_answer_span(answer_text: str, gaze_tokens: Sequence[str]) -> tuple[int, int] | None:
    answer_norm_tokens = normalized_answer_tokens(answer_text)
    if not answer_norm_tokens:
        return None

    gaze_norm_tokens = [normalize_token_for_alignment(token) for token in gaze_tokens]
    gaze_norm_tokens = [token for token in gaze_norm_tokens]
    answer_joined = "".join(answer_norm_tokens)
    for start in range(len(gaze_norm_tokens)):
        current: list[str] = []
        for end in range(start, len(gaze_norm_tokens)):
            token = gaze_norm_tokens[end]
            if not token:
                continue
            current.append(token)
            current_joined = "".join(current)
            if current_joined == answer_joined:
                return start, end
            if not answer_joined.startswith(current_joined):
                break
    return None


def normalized_answer_tokens(text: str) -> list[str]:
    repaired = repair_mojibake(text)
    tokens = re.findall(r"\w+|[^\w\s]", repaired, flags=re.UNICODE)
    return [normalize_token_for_alignment(token) for token in tokens if normalize_token_for_alignment(token)]


def score_answer(gold_answer: str, predicted_answer: str) -> AnswerScores:
    normalized_gold = normalize_answer(gold_answer)
    normalized_predicted = normalize_answer(predicted_answer)
    exact_match = normalized_gold == normalized_predicted and normalized_gold != ""
    gold_tokens = normalized_gold.split()
    predicted_tokens = normalized_predicted.split()
    if not gold_tokens or not predicted_tokens:
        return AnswerScores(
            exact_match=exact_match,
            token_f1=0.0,
            precision=0.0,
            recall=0.0,
            normalized_gold_answer=normalized_gold,
            normalized_predicted_answer=normalized_predicted,
        )

    common = Counter(gold_tokens) & Counter(predicted_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        precision = 0.0
        recall = 0.0
        token_f1 = 0.0
    else:
        precision = overlap / len(predicted_tokens)
        recall = overlap / len(gold_tokens)
        token_f1 = 2.0 * precision * recall / (precision + recall)
    return AnswerScores(
        exact_match=exact_match,
        token_f1=float(token_f1),
        precision=float(precision),
        recall=float(recall),
        normalized_gold_answer=normalized_gold,
        normalized_predicted_answer=normalized_predicted,
    )


def normalize_answer(text: str) -> str:
    text = repair_mojibake(text).lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def correlation_bundle(x_values: Sequence[Any], y_values: Sequence[Any]) -> dict[str, object]:
    frame = pd.DataFrame({"x": x_values, "y": y_values}).replace([np.inf, -np.inf], np.nan).dropna()
    n = int(len(frame))
    if n < 3 or frame["x"].nunique() < 2 or frame["y"].nunique() < 2:
        return {
            "n": n,
            "pearson_r": math.nan,
            "pearson_p": math.nan,
            "spearman_rho": math.nan,
            "spearman_p": math.nan,
            "status": "not_computed",
        }
    pearson = stats.pearsonr(frame["x"], frame["y"])
    spearman = stats.spearmanr(frame["x"], frame["y"], nan_policy="omit")
    return {
        "n": n,
        "pearson_r": float(pearson.statistic),
        "pearson_p": float(pearson.pvalue),
        "spearman_rho": float(spearman.statistic),
        "spearman_p": float(spearman.pvalue),
        "status": "ok",
    }


def write_report(
    path: Path,
    correctness: pd.DataFrame,
    correlations: pd.DataFrame,
    spans: pd.DataFrame,
    summary: pd.DataFrame,
) -> None:
    report = {
        "text_count": int(len(summary)),
        "exact_match_count": int(correctness.get("answer_exact_match", pd.Series(dtype=bool)).sum()),
        "mean_answer_token_f1": safe_mean(correctness.get("answer_token_f1")),
        "gold_span_found_count": int(
            (spans[(spans.get("span_type") == "gold")].get("span_found", pd.Series(dtype=bool))).sum()
        )
        if not spans.empty
        else 0,
        "predicted_span_found_count": int(
            (spans[(spans.get("span_type") == "predicted")].get("span_found", pd.Series(dtype=bool))).sum()
        )
        if not spans.empty
        else 0,
        "feature_ablation_correlation_rows": int(len(correlations)),
        "outputs": {
            "answer_correctness_csv": str(path.parent / "answer_correctness.csv"),
            "feature_ablation_gaze_correlations_csv": str(
                path.parent / "feature_ablation_gaze_correlations.csv"
            ),
            "answer_span_summaries_csv": str(path.parent / "answer_span_summaries.csv"),
            "relationship_summary_csv": str(path.parent / "relationship_summary.csv"),
        },
    }
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def qa_display_canonical_key(payload: dict[str, Any]) -> str:
    context = str(payload.get("context", "")).strip()
    question = str(payload.get("question", "")).strip()
    if not context or not question:
        raise ValueError("QA result missing non-empty context/question")
    return canonical_text_key(format_display_text(context, question))


def format_display_text(context: str, question: str) -> str:
    return "\n".join(["Context:", context, "", "Question:", question])


def canonical_text_key(text: str) -> str:
    normalized = str(text).lower().replace("\ufeff", "")
    return re.sub(r"\s+", "", normalized)


def repair_mojibake(token: str) -> str:
    candidates = [str(token)]
    for encoding in ("cp1250", "cp1252", "latin1"):
        try:
            candidates.append(str(token).encode(encoding).decode("utf-8"))
        except UnicodeError:
            continue
    return min(candidates, key=mojibake_score)


def mojibake_score(token: str) -> int:
    markers = (chr(0xFFFD), chr(0x00C2), chr(0x00C3), chr(0x00C4), chr(0x00C5))
    return sum(str(token).count(marker) for marker in markers)


def load_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root is not an object: {path}")
    return payload


def safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def safe_mean(values: Any) -> float:
    if values is None:
        return math.nan
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    return float(numeric.mean()) if len(numeric) else math.nan
