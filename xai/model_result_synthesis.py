import html
import json
import math
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns

matplotlib.use("Agg")
import matplotlib.pyplot as plt


MODEL_ONLY_SPECS: tuple[tuple[str, str, str, str], ...] = (
    ("bert_masking", "bert_masking", "aligned_representation.csv", "bert_masking"),
    (
        "bert_masking_qa_context",
        "bert_masking_qa_context",
        "aligned_representation.csv",
        "bert_masking",
    ),
    ("nli", "nli", "aligned_decision.csv", "nli"),
    ("nli_qa_context", "nli_qa_context", "aligned_decision.csv", "nli"),
    ("surprisal", "surprisal", "aligned_surprisal.csv", "surprisal"),
    ("surprisal_pl", "surprisal_pl", "aligned_surprisal.csv", "surprisal"),
    (
        "surprisal_qa_context",
        "surprisal_qa_context",
        "aligned_surprisal.csv",
        "surprisal",
    ),
    (
        "surprisal_pl_qa_context",
        "surprisal_pl_qa_context",
        "aligned_surprisal.csv",
        "surprisal",
    ),
)

ATTN_DIST_METRICS: tuple[str, ...] = (
    "normentw",
    "normentn",
    "normentr",
    "deltaentw",
    "deltaentn",
    "deltaentr",
    "manhattanw",
    "manhattann",
    "manhattanr",
    "emdw",
    "emdn",
    "emdr",
    "surprisal",
    "nli",
    "masking",
)

QA_GAZE_METRICS: tuple[str, ...] = (
    "qa_lig",
    "qa_lig_start",
    "qa_lig_end",
    "qa_feature_ablation",
    "qa_feature_ablation_start",
    "qa_feature_ablation_end",
)

PRIMARY_METRICS: tuple[str, ...] = (
    "bert_masking",
    "surprisal",
    "normentr",
    "manhattanr",
    "qa_lig",
    "qa_feature_ablation",
)

GAZE_METRIC_ORDER: tuple[str, ...] = (
    "fixation_event_dwell_time_mean",
    "fixation_event_dwell_time_median",
    "fixation_event_dwell_time_sum",
    "fixation_event_dwell_time_nonzero_rate",
    "dwell_time_mean",
    "dwell_time_median",
    "dwell_time_sum",
    "dwell_time_nonzero_rate",
    "fixation_event_count_mean",
    "fixation_event_count_sum",
)

METRIC_LABELS: dict[str, str] = {
    "bert_masking": "BERT masking",
    "surprisal": "Surprisal",
    "normentw": "Attention entropy W",
    "normentn": "Attention entropy N",
    "normentr": "Attention entropy R",
    "deltaentw": "Delta entropy W",
    "deltaentn": "Delta entropy N",
    "deltaentr": "Delta entropy R",
    "manhattanw": "Manhattan distance W",
    "manhattann": "Manhattan distance N",
    "manhattanr": "Manhattan distance R",
    "emdw": "Earth mover distance W",
    "emdn": "Earth mover distance N",
    "emdr": "Earth mover distance R",
    "nli": "NLI attribution",
    "masking": "Masking score",
    "qa_lig": "QA LIG",
    "qa_lig_start": "QA LIG start",
    "qa_lig_end": "QA LIG end",
    "qa_feature_ablation": "QA feature ablation",
    "qa_feature_ablation_start": "QA feature ablation start",
    "qa_feature_ablation_end": "QA feature ablation end",
}


@dataclass(frozen=True)
class TextManifestEntry:
    text_id: str
    language: str
    preview: str
    slug_key: str
    tokens: tuple[str, ...]


@dataclass(frozen=True)
class ModelArtifact:
    family: str
    metric: str
    path: Path
    artifact_id: str
    model_name: str
    text_id: str
    language: str
    text_preview: str


@dataclass(frozen=True)
class SynthesisResult:
    output_dir: Path
    run_summary_path: Path
    family_summary_path: Path
    top_tokens_path: Path
    per_method_summary_path: Path
    per_method_text_summary_path: Path
    qa_gaze_correlation_summary_path: Path
    qa_answer_synthesis_path: Path
    html_report_path: Path
    plot_paths: tuple[Path, ...]


class ModelResultSynthesizer:
    def __init__(self, results_dir: Path, output_dir: Path, top_token_count: int = 10) -> None:
        self.results_dir = results_dir
        self.output_dir = output_dir
        self.top_token_count = top_token_count
        self.plots_dir = output_dir / "plots"
        self.per_method_dir = output_dir / "per_method"

    def run(self) -> SynthesisResult:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        self.per_method_dir.mkdir(parents=True, exist_ok=True)

        manifest = load_text_manifest(self.results_dir / "eye_tracking_aggregates" / "text_aggregate_manifest.csv")
        token_scores = self._load_all_token_scores(manifest)
        run_summary = build_run_summary(token_scores)
        family_summary = build_family_summary(token_scores)
        top_tokens = build_top_tokens(token_scores, self.top_token_count)
        qa_gaze_correlations = load_qa_gaze_correlation_summary(self.results_dir / "qa_gaze_benchmark")
        qa_answer_synthesis = load_qa_answer_synthesis(self.results_dir / "qa_answer_gaze_relationship")
        per_method_summary = build_per_method_summary(
            family_summary=family_summary,
            run_summary=run_summary,
            top_tokens=top_tokens,
            qa_gaze_correlations=qa_gaze_correlations,
        )
        per_method_text_summary = build_per_method_text_summary(run_summary)

        run_summary_path = self.output_dir / "model_run_summary.csv"
        family_summary_path = self.output_dir / "model_family_summary.csv"
        top_tokens_path = self.output_dir / "model_top_tokens.csv"
        per_method_summary_path = self.per_method_dir / "per_method_clean_summary.csv"
        per_method_text_summary_path = self.per_method_dir / "per_method_text_summary.csv"
        qa_gaze_correlation_summary_path = self.output_dir / "qa_gaze_correlation_summary.csv"
        qa_answer_synthesis_path = self.output_dir / "qa_answer_synthesis.csv"

        run_summary.to_csv(run_summary_path, index=False)
        family_summary.to_csv(family_summary_path, index=False)
        top_tokens.to_csv(top_tokens_path, index=False)
        per_method_summary.to_csv(per_method_summary_path, index=False)
        per_method_text_summary.to_csv(per_method_text_summary_path, index=False)
        qa_gaze_correlations.to_csv(qa_gaze_correlation_summary_path, index=False)
        qa_answer_synthesis.to_csv(qa_answer_synthesis_path, index=False)

        plot_paths = tuple(
            path
            for path in [
                save_family_metric_plot(family_summary, self.plots_dir / "family_metric_mean_abs.png"),
                save_text_metric_heatmap(run_summary, self.plots_dir / "text_metric_mean_abs_heatmap.png"),
                save_qa_answer_plot(qa_answer_synthesis, self.plots_dir / "qa_answer_synthesis.png"),
                save_qa_gaze_correlation_plot(
                    qa_gaze_correlations,
                    self.plots_dir / "qa_gaze_correlation_overview.png",
                ),
            ]
            if path is not None
        )
        per_method_plot_paths = tuple(save_per_method_plots(run_summary, top_tokens, self.per_method_dir))
        all_plot_paths = (*plot_paths, *per_method_plot_paths)

        html_report_path = self.output_dir / "model_synthesis_report.html"
        save_html_report(
            html_report_path=html_report_path,
            run_summary=run_summary,
            family_summary=family_summary,
            per_method_summary=per_method_summary,
            top_tokens=top_tokens,
            qa_gaze_correlations=qa_gaze_correlations,
            qa_answer_synthesis=qa_answer_synthesis,
            plot_paths=plot_paths,
            per_method_plot_paths=per_method_plot_paths,
        )

        return SynthesisResult(
            output_dir=self.output_dir,
            run_summary_path=run_summary_path,
            family_summary_path=family_summary_path,
            top_tokens_path=top_tokens_path,
            per_method_summary_path=per_method_summary_path,
            per_method_text_summary_path=per_method_text_summary_path,
            qa_gaze_correlation_summary_path=qa_gaze_correlation_summary_path,
            qa_answer_synthesis_path=qa_answer_synthesis_path,
            html_report_path=html_report_path,
            plot_paths=all_plot_paths,
        )

    def _load_all_token_scores(self, manifest: Mapping[str, TextManifestEntry]) -> pd.DataFrame:
        frames: list[pd.DataFrame] = []
        for family, directory, filename, metric in MODEL_ONLY_SPECS:
            for path in sorted((self.results_dir / directory).rglob(filename)):
                artifact = build_model_artifact(
                    family=family,
                    metric=metric,
                    path=path,
                    results_dir=self.results_dir,
                    manifest=manifest,
                )
                frames.append(load_token_score_frame(artifact))

        for path in sorted((self.results_dir / "attn_dist").glob("*.csv")):
            text_id = infer_text_id_from_attention_dist_csv(path, manifest)
            manifest_entry = manifest.get(text_id)
            language = manifest_entry.language if manifest_entry else ""
            preview = manifest_entry.preview if manifest_entry else ""
            frame = pd.read_csv(path, sep=";", engine="python")
            for metric in ATTN_DIST_METRICS:
                if metric not in frame.columns:
                    continue
                selected = frame.loc[:, ["token", metric]].rename(columns={metric: "score"})
                selected["family"] = "attn_dist"
                selected["metric"] = metric
                selected["metric_label"] = METRIC_LABELS.get(metric, metric)
                selected["artifact_id"] = path.stem
                selected["model_name"] = "attention-distance export"
                selected["text_id"] = text_id
                selected["language"] = language
                selected["text_preview"] = preview
                selected["source_path"] = str(path)
                frames.append(prepare_token_scores(selected))

        qa_dir = self.results_dir / "qa_gaze_benchmark"
        for path in sorted(qa_dir.rglob("aligned_tokens.csv")):
            text_id = path.parent.name
            manifest_entry = manifest.get(text_id)
            language = manifest_entry.language if manifest_entry else ""
            preview = manifest_entry.preview if manifest_entry else ""
            frame = pd.read_csv(path)
            for metric in QA_GAZE_METRICS:
                if metric not in frame.columns:
                    continue
                token_column = "gaze_token" if "gaze_token" in frame.columns else "attention_token"
                selected = frame.loc[:, [token_column, metric]].rename(columns={token_column: "token", metric: "score"})
                selected["family"] = "qa_gaze_benchmark"
                selected["metric"] = metric
                selected["metric_label"] = METRIC_LABELS.get(metric, metric)
                selected["artifact_id"] = text_id
                selected["model_name"] = "henryk/bert-base-multilingual-cased-finetuned-polish-squad2"
                selected["text_id"] = text_id
                selected["language"] = language
                selected["text_preview"] = preview
                selected["source_path"] = str(path)
                frames.append(prepare_token_scores(selected))

        if not frames:
            return empty_token_scores()
        return pd.concat(frames, ignore_index=True)


def load_text_manifest(path: Path) -> dict[str, TextManifestEntry]:
    frame = pd.read_csv(path)
    entries: dict[str, TextManifestEntry] = {}
    for row in frame.itertuples(index=False):
        csv_path = Path(str(row.csv_path))
        stem = csv_path.stem
        stem = re.sub(r"^\d+_", "", stem)
        stem = re.sub(r"_[0-9a-f]{12}$", "", stem)
        text_id = str(row.text_id)
        entries[text_id] = TextManifestEntry(
            text_id=text_id,
            language=str(row.text_language),
            preview=str(row.text_preview),
            slug_key=slug_key(stem),
            tokens=load_manifest_tokens(csv_path),
        )
    return entries


def load_manifest_tokens(path: Path) -> tuple[str, ...]:
    try:
        frame = pd.read_csv(path)
    except (OSError, UnicodeDecodeError, pd.errors.ParserError):
        return ()
    if "token" not in frame.columns:
        return ()
    return tuple(frame["token"].astype(str).tolist())


def build_model_artifact(
    family: str,
    metric: str,
    path: Path,
    results_dir: Path,
    manifest: Mapping[str, TextManifestEntry],
) -> ModelArtifact:
    artifact_id = str(path.parent.relative_to(results_dir / family))
    text_id = infer_text_id_from_path(path, manifest)
    manifest_entry = manifest.get(text_id)
    language = manifest_entry.language if manifest_entry else ""
    preview = manifest_entry.preview if manifest_entry else ""
    model_name = infer_model_name(path, metric)
    return ModelArtifact(
        family=family,
        metric=metric,
        path=path,
        artifact_id=artifact_id,
        model_name=model_name,
        text_id=text_id,
        language=language,
        text_preview=preview,
    )


def infer_text_id_from_path(path: Path, manifest: Mapping[str, TextManifestEntry]) -> str:
    payload_slug = read_pipeline_text_slug(path.parent / "pipeline_result.json")
    candidate_slugs = [
        payload_slug,
        strip_timestamp_prefix(path.parent.name),
        path.parent.name,
        path.parent.parent.name,
    ]
    candidate_keys = [slug_key(candidate) for candidate in candidate_slugs if candidate]
    best_text_id = ""
    best_score = 0
    for text_id, entry in manifest.items():
        for candidate_key in candidate_keys:
            if not candidate_key or not entry.slug_key:
                continue
            score = overlap_score(candidate_key, entry.slug_key)
            if score > best_score:
                best_text_id = text_id
                best_score = score
    return best_text_id


def infer_text_id_from_attention_dist_csv(path: Path, manifest: Mapping[str, TextManifestEntry]) -> str:
    try:
        frame = pd.read_csv(path, sep=";", engine="python")
    except (OSError, UnicodeDecodeError, pd.errors.ParserError):
        return ""
    if "token" not in frame.columns:
        return ""

    tokens = tuple(frame["token"].astype(str).tolist())
    if not tokens:
        return ""

    best_text_id = ""
    best_score = -1
    for text_id, entry in manifest.items():
        score = ordered_token_overlap_score(tokens, entry.tokens)
        if score > best_score:
            best_text_id = text_id
            best_score = score
    return best_text_id


def ordered_token_overlap_score(candidate_tokens: Sequence[str], manifest_tokens: Sequence[str]) -> int:
    normalized_manifest = [normalize_token_for_overlap(token) for token in manifest_tokens]
    normalized_candidate = [normalize_token_for_overlap(token) for token in candidate_tokens]
    normalized_manifest = [token for token in normalized_manifest if token]
    normalized_candidate = [token for token in normalized_candidate if token]
    if not normalized_candidate or not normalized_manifest:
        return 0

    best_score = 0
    max_offset = max(0, len(normalized_manifest) - len(normalized_candidate))
    for offset in range(max_offset + 1):
        score = 0
        for index, token in enumerate(normalized_candidate):
            manifest_index = offset + index
            if manifest_index >= len(normalized_manifest):
                break
            if token == normalized_manifest[manifest_index]:
                score += 1
        best_score = max(best_score, score)

    manifest_set = set(normalized_manifest)
    set_score = sum(1 for token in normalized_candidate if token in manifest_set)
    return max(best_score, set_score)


def read_pipeline_text_slug(path: Path) -> str:
    payload = read_json(path)
    return str(payload.get("text_slug", ""))


def infer_model_name(path: Path, metric: str) -> str:
    if metric == "surprisal":
        surprisal_json = path.with_name("surprisal.json")
        payload = read_json(surprisal_json)
        model_name = str(payload.get("model_name") or payload.get("model_key") or "")
        if model_name:
            return model_name

    pipeline_result_path = path.with_name("pipeline_result.json")
    config = read_json(pipeline_result_path).get("config", {})
    if isinstance(config, dict):
        if metric == "bert_masking":
            return str(config.get("bert_model_name") or "")
        if metric == "nli":
            return str(config.get("nli_model_name") or "")
        if metric == "surprisal":
            return str(config.get("surprisal_model") or "")
    return ""


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def load_token_score_frame(artifact: ModelArtifact) -> pd.DataFrame:
    frame = pd.read_csv(artifact.path)
    if "token" not in frame.columns or artifact.metric not in frame.columns:
        raise ValueError(f"{artifact.path} missing token/{artifact.metric} columns")
    selected = frame.loc[:, ["token", artifact.metric]].rename(columns={artifact.metric: "score"})
    selected["family"] = artifact.family
    selected["metric"] = artifact.metric
    selected["metric_label"] = METRIC_LABELS.get(artifact.metric, artifact.metric)
    selected["artifact_id"] = artifact.artifact_id
    selected["model_name"] = artifact.model_name
    selected["text_id"] = artifact.text_id
    selected["language"] = artifact.language
    selected["text_preview"] = artifact.text_preview
    selected["source_path"] = str(artifact.path)
    return prepare_token_scores(selected)


def prepare_token_scores(frame: pd.DataFrame) -> pd.DataFrame:
    prepared = frame.copy()
    prepared["token"] = prepared["token"].astype(str).str.strip()
    prepared["score"] = pd.to_numeric(prepared["score"], errors="coerce")
    prepared = prepared.replace([np.inf, -np.inf], np.nan).dropna(subset=["score"])
    prepared = prepared[prepared["token"].ne("")].copy()
    prepared["abs_score"] = prepared["score"].abs()
    prepared["positive_score"] = prepared["score"].gt(0)
    return prepared.loc[
        :,
        [
            "family",
            "metric",
            "metric_label",
            "artifact_id",
            "model_name",
            "text_id",
            "language",
            "text_preview",
            "token",
            "score",
            "abs_score",
            "positive_score",
            "source_path",
        ],
    ]


def empty_token_scores() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "family",
            "metric",
            "metric_label",
            "artifact_id",
            "model_name",
            "text_id",
            "language",
            "text_preview",
            "token",
            "score",
            "abs_score",
            "positive_score",
            "source_path",
        ]
    )


def build_run_summary(token_scores: pd.DataFrame) -> pd.DataFrame:
    if token_scores.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    group_columns = [
        "family",
        "metric",
        "metric_label",
        "artifact_id",
        "model_name",
        "text_id",
        "language",
        "text_preview",
        "source_path",
    ]
    for keys, group in token_scores.groupby(group_columns, dropna=False, sort=True):
        key_data = dict(zip(group_columns, keys))
        top_abs = group.loc[group["abs_score"].idxmax()]
        top_positive = group.loc[group["score"].idxmax()]
        top_negative = group.loc[group["score"].idxmin()]
        rows.append(
            {
                **key_data,
                "token_n": int(len(group)),
                "mean": float(group["score"].mean()),
                "std": safe_float(group["score"].std()),
                "median": float(group["score"].median()),
                "min": float(group["score"].min()),
                "max": float(group["score"].max()),
                "mean_abs": float(group["abs_score"].mean()),
                "median_abs": float(group["abs_score"].median()),
                "max_abs": float(group["abs_score"].max()),
                "positive_rate": float(group["positive_score"].mean()),
                "top_abs_token": str(top_abs["token"]),
                "top_abs_score": float(top_abs["score"]),
                "top_positive_token": str(top_positive["token"]),
                "top_positive_score": float(top_positive["score"]),
                "top_negative_token": str(top_negative["token"]),
                "top_negative_score": float(top_negative["score"]),
            }
        )
    return pd.DataFrame(rows)


def build_family_summary(token_scores: pd.DataFrame) -> pd.DataFrame:
    if token_scores.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for (family, metric, metric_label, model_name), group in token_scores.groupby(
        ["family", "metric", "metric_label", "model_name"],
        dropna=False,
        sort=True,
    ):
        rows.append(
            {
                "family": family,
                "metric": metric,
                "metric_label": metric_label,
                "model_name": model_name,
                "run_n": int(group["artifact_id"].nunique()),
                "text_n": int(group["text_id"].nunique()),
                "token_observation_n": int(len(group)),
                "mean": float(group["score"].mean()),
                "std": safe_float(group["score"].std()),
                "median": float(group["score"].median()),
                "min": float(group["score"].min()),
                "max": float(group["score"].max()),
                "mean_abs": float(group["abs_score"].mean()),
                "median_abs": float(group["abs_score"].median()),
                "max_abs": float(group["abs_score"].max()),
                "positive_rate": float(group["positive_score"].mean()),
            }
        )
    return pd.DataFrame(rows)


def build_top_tokens(token_scores: pd.DataFrame, top_token_count: int) -> pd.DataFrame:
    if token_scores.empty:
        return pd.DataFrame()
    ranked = token_scores.copy()
    ranked["rank_abs_within_run_metric"] = (
        ranked.groupby(["family", "metric", "artifact_id"], sort=True)["abs_score"]
        .rank(method="first", ascending=False)
        .astype(int)
    )
    ranked = ranked[ranked["rank_abs_within_run_metric"].le(top_token_count)].copy()
    return ranked.sort_values(["family", "metric", "artifact_id", "rank_abs_within_run_metric"])


def build_per_method_summary(
    family_summary: pd.DataFrame,
    run_summary: pd.DataFrame,
    top_tokens: pd.DataFrame,
    qa_gaze_correlations: pd.DataFrame,
) -> pd.DataFrame:
    output_columns = [
        "method_id",
        "family",
        "metric",
        "metric_label",
        "model_name",
        "run_n",
        "text_n",
        "token_observation_n",
        "mean_abs_score",
        "median_abs_score",
        "max_abs_score",
        "strongest_text_id",
        "strongest_text_language",
        "strongest_text_mean_abs",
        "strongest_text_top_token",
        "strongest_token",
        "strongest_token_text_id",
        "strongest_token_score",
        "strongest_gaze_metric",
        "strongest_gaze_mean_pearson_r",
        "strongest_gaze_mean_abs_pearson_r",
        "strongest_gaze_pearson_fdr_lt_05_n",
        "strongest_gaze_spearman_fdr_lt_05_n",
    ]
    if family_summary.empty:
        return pd.DataFrame(columns=output_columns)

    rows: list[dict[str, object]] = []
    for method in family_summary.itertuples(index=False):
        family = str(method.family)
        metric = str(method.metric)
        method_runs = run_summary[run_summary["family"].eq(family) & run_summary["metric"].eq(metric)]
        method_tokens = top_tokens[top_tokens["family"].eq(family) & top_tokens["metric"].eq(metric)]
        strongest_run = method_runs.sort_values("mean_abs", ascending=False).head(1)
        strongest_token = method_tokens.sort_values("abs_score", ascending=False).head(1)
        strongest_gaze = strongest_qa_gaze_row(qa_gaze_correlations, metric)

        rows.append(
            {
                "method_id": method_identifier(family, metric),
                "family": family,
                "metric": metric,
                "metric_label": method.metric_label,
                "model_name": method.model_name,
                "run_n": method.run_n,
                "text_n": method.text_n,
                "token_observation_n": method.token_observation_n,
                "mean_abs_score": method.mean_abs,
                "median_abs_score": method.median_abs,
                "max_abs_score": method.max_abs,
                "strongest_text_id": value_from_first(strongest_run, "text_id"),
                "strongest_text_language": value_from_first(strongest_run, "language"),
                "strongest_text_mean_abs": value_from_first(strongest_run, "mean_abs"),
                "strongest_text_top_token": value_from_first(strongest_run, "top_abs_token"),
                "strongest_token": value_from_first(strongest_token, "token"),
                "strongest_token_text_id": value_from_first(strongest_token, "text_id"),
                "strongest_token_score": value_from_first(strongest_token, "score"),
                "strongest_gaze_metric": value_from_first(strongest_gaze, "gaze_metric"),
                "strongest_gaze_mean_pearson_r": value_from_first(strongest_gaze, "mean_pearson_r"),
                "strongest_gaze_mean_abs_pearson_r": value_from_first(strongest_gaze, "mean_abs_pearson_r"),
                "strongest_gaze_pearson_fdr_lt_05_n": value_from_first(strongest_gaze, "pearson_fdr_lt_05_n"),
                "strongest_gaze_spearman_fdr_lt_05_n": value_from_first(strongest_gaze, "spearman_fdr_lt_05_n"),
            }
        )

    return pd.DataFrame(rows, columns=output_columns)


def build_per_method_text_summary(run_summary: pd.DataFrame) -> pd.DataFrame:
    output_columns = [
        "method_id",
        "family",
        "metric",
        "metric_label",
        "text_id",
        "language",
        "token_n",
        "mean_abs_score",
        "median_abs_score",
        "max_abs_score",
        "top_abs_token",
        "top_abs_score",
        "top_positive_token",
        "top_positive_score",
        "top_negative_token",
        "top_negative_score",
        "text_preview",
    ]
    if run_summary.empty:
        return pd.DataFrame(columns=output_columns)

    text_summary = run_summary.copy()
    text_summary["method_id"] = text_summary.apply(
        lambda row: method_identifier(str(row["family"]), str(row["metric"])),
        axis=1,
    )
    text_summary = text_summary.rename(
        columns={
            "mean_abs": "mean_abs_score",
            "median_abs": "median_abs_score",
            "max_abs": "max_abs_score",
        }
    )
    return text_summary.loc[:, output_columns].sort_values(["method_id", "mean_abs_score"], ascending=[True, False])


def strongest_qa_gaze_row(qa_gaze_correlations: pd.DataFrame, metric: str) -> pd.DataFrame:
    if qa_gaze_correlations.empty or "attention_metric" not in qa_gaze_correlations.columns:
        return pd.DataFrame()
    selected = qa_gaze_correlations[qa_gaze_correlations["attention_metric"].eq(metric)].copy()
    if selected.empty:
        return selected
    return selected.sort_values("mean_abs_pearson_r", ascending=False).head(1)


def value_from_first(frame: pd.DataFrame, column: str) -> object:
    if frame.empty or column not in frame.columns:
        return ""
    value = frame.iloc[0][column]
    if pd.isna(value):
        return ""
    return value


def load_qa_gaze_correlation_summary(qa_gaze_dir: Path) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for path in sorted(qa_gaze_dir.rglob("correlation_statistics.csv")):
        frame = pd.read_csv(path)
        frame["text_id"] = path.parent.name
        rows.append(frame)
    if not rows:
        return pd.DataFrame()
    combined = pd.concat(rows, ignore_index=True)
    combined = combined[combined["status"].eq("ok")].copy()
    combined["abs_pearson_r"] = combined["pearson_r"].abs()
    combined["abs_spearman_rho"] = combined["spearman_rho"].abs()
    summary = (
        combined.groupby(["attention_metric", "gaze_metric"], sort=True)
        .agg(
            text_n=("text_id", "nunique"),
            row_n=("text_id", "count"),
            mean_pearson_r=("pearson_r", "mean"),
            median_pearson_r=("pearson_r", "median"),
            mean_abs_pearson_r=("abs_pearson_r", "mean"),
            max_abs_pearson_r=("abs_pearson_r", "max"),
            mean_spearman_rho=("spearman_rho", "mean"),
            median_spearman_rho=("spearman_rho", "median"),
            mean_abs_spearman_rho=("abs_spearman_rho", "mean"),
            max_abs_spearman_rho=("abs_spearman_rho", "max"),
            pearson_fdr_lt_05_n=("pearson_fdr_q", lambda values: int((values < 0.05).sum())),
            spearman_fdr_lt_05_n=("spearman_fdr_q", lambda values: int((values < 0.05).sum())),
        )
        .reset_index()
    )
    return summary


def load_qa_answer_synthesis(qa_answer_dir: Path) -> pd.DataFrame:
    correctness_path = qa_answer_dir / "answer_correctness.csv"
    relationship_path = qa_answer_dir / "relationship_summary.csv"
    if not correctness_path.exists():
        return pd.DataFrame()
    correctness = pd.read_csv(correctness_path)
    if relationship_path.exists():
        relationship = pd.read_csv(relationship_path)
        keep_columns = [
            "text_id",
            "strongest_gaze_metric",
            "strongest_pearson_r",
            "strongest_spearman_rho",
            "gold_span_found",
            "predicted_span_found",
            "gold_qa_feature_ablation_abs_sum",
            "predicted_qa_feature_ablation_abs_sum",
            "gold_dwell_time_mean",
            "predicted_dwell_time_mean",
            "gold_fixation_event_count_mean",
            "predicted_fixation_event_count_mean",
        ]
        relationship = relationship.loc[:, [column for column in keep_columns if column in relationship.columns]]
        return correctness.merge(relationship, on="text_id", how="left", suffixes=("", "_relationship"))
    return correctness


def save_family_metric_plot(summary: pd.DataFrame, path: Path) -> Path | None:
    if summary.empty:
        return None
    plot_frame = summary.copy()
    plot_frame["label"] = plot_frame["family"] + " / " + plot_frame["metric_label"]
    plot_frame = plot_frame.sort_values("mean_abs", ascending=True)
    fig, axis = plt.subplots(figsize=(12, max(5, len(plot_frame) * 0.42)))
    axis.barh(plot_frame["label"], plot_frame["mean_abs"], color="#2563eb", alpha=0.86)
    axis.set_xlabel("Mean absolute model score")
    axis.set_title("Mean absolute score by model family and metric")
    axis.grid(axis="x", alpha=0.25, linestyle="--")
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


def save_text_metric_heatmap(run_summary: pd.DataFrame, path: Path) -> Path | None:
    if run_summary.empty:
        return None
    plot_frame = run_summary[run_summary["metric"].isin(PRIMARY_METRICS)].copy()
    if plot_frame.empty:
        return None
    plot_frame["row_label"] = plot_frame.apply(
        lambda row: f"{row['text_id']} ({row['language']})",
        axis=1,
    )
    plot_frame["metric_key"] = plot_frame["family"] + " / " + plot_frame["metric_label"]
    pivot = plot_frame.pivot_table(index="row_label", columns="metric_key", values="mean_abs", aggfunc="mean")
    fig, axis = plt.subplots(figsize=(12, max(5, len(pivot) * 0.42)))
    sns.heatmap(pivot, annot=True, fmt=".3g", cmap="viridis", linewidths=0.5, ax=axis)
    axis.set_title("Mean absolute model score by text")
    axis.set_xlabel("")
    axis.set_ylabel("")
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


def save_qa_answer_plot(frame: pd.DataFrame, path: Path) -> Path | None:
    if frame.empty:
        return None
    plot_frame = frame.copy()
    plot_frame["label"] = plot_frame["text_id"] + " (" + plot_frame["language"].astype(str) + ")"
    plot_frame = plot_frame.sort_values(["answer_exact_match", "answer_token_f1", "span_score"], ascending=True)
    y_pos = np.arange(len(plot_frame))
    fig, axes = plt.subplots(1, 2, figsize=(14, max(5, len(plot_frame) * 0.45)), sharey=True)
    axes[0].barh(y_pos, pd.to_numeric(plot_frame["answer_token_f1"], errors="coerce"), color="#2563eb", alpha=0.86)
    axes[0].set_title("Answer token F1")
    axes[0].set_xlabel("F1")
    axes[0].grid(axis="x", alpha=0.25, linestyle="--")
    axes[1].barh(y_pos, pd.to_numeric(plot_frame["span_score"], errors="coerce"), color="#d97706", alpha=0.86)
    axes[1].set_title("Predicted span score")
    axes[1].set_xlabel("Start probability x end probability")
    axes[1].grid(axis="x", alpha=0.25, linestyle="--")
    axes[0].set_yticks(y_pos)
    axes[0].set_yticklabels(plot_frame["label"], fontsize=8)
    fig.suptitle("QA answer quality by text", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


def save_qa_gaze_correlation_plot(summary: pd.DataFrame, path: Path) -> Path | None:
    if summary.empty:
        return None
    plot_frame = summary[summary["gaze_metric"].isin(GAZE_METRIC_ORDER)].copy()
    if plot_frame.empty:
        return None
    pivot = plot_frame.pivot_table(
        index="gaze_metric",
        columns="attention_metric",
        values="mean_pearson_r",
        aggfunc="mean",
    ).reindex(GAZE_METRIC_ORDER)
    fig, axis = plt.subplots(figsize=(12, 7))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="coolwarm", center=0.0, linewidths=0.5, ax=axis)
    axis.set_title("Mean Pearson correlation between QA attribution metrics and gaze")
    axis.set_xlabel("QA metric")
    axis.set_ylabel("Gaze metric")
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


def save_per_method_plots(run_summary: pd.DataFrame, top_tokens: pd.DataFrame, output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    if run_summary.empty:
        return []

    plot_paths: list[Path] = []
    for (family, metric), method_runs in run_summary.groupby(["family", "metric"], sort=True):
        method_id = method_identifier(str(family), str(metric))
        method_tokens = top_tokens[top_tokens["family"].eq(family) & top_tokens["metric"].eq(metric)]
        text_plot_path = save_per_method_text_score_plot(
            method_runs,
            output_dir / f"{method_id}_text_scores.png",
        )
        token_plot_path = save_per_method_top_token_plot(
            method_tokens,
            output_dir / f"{method_id}_top_tokens.png",
        )
        if text_plot_path is not None:
            plot_paths.append(text_plot_path)
        if token_plot_path is not None:
            plot_paths.append(token_plot_path)
    return plot_paths


def save_per_method_text_score_plot(method_runs: pd.DataFrame, path: Path) -> Path | None:
    if method_runs.empty:
        return None
    plot_frame = method_runs.sort_values("mean_abs", ascending=True).copy()
    plot_frame["label"] = plot_frame.apply(
        lambda row: f"{row['text_id']} ({row['language']})",
        axis=1,
    )
    method_label = str(plot_frame.iloc[0]["metric_label"])
    y_pos = np.arange(len(plot_frame))
    fig, axis = plt.subplots(figsize=(11, max(4, len(plot_frame) * 0.5)))
    axis.barh(y_pos, plot_frame["mean_abs"], color="#2563eb", alpha=0.86)
    axis.set_yticks(y_pos)
    axis.set_yticklabels(plot_frame["label"], fontsize=8)
    axis.set_xlabel("Mean absolute score")
    axis.set_title(f"{method_label}: text-level score profile")
    axis.grid(axis="x", alpha=0.25, linestyle="--")

    for index, row in enumerate(plot_frame.itertuples(index=False)):
        axis.text(
            float(row.mean_abs),
            index,
            f"  {short_token_label(str(row.top_abs_token), 18)}",
            va="center",
            fontsize=8,
        )

    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


def save_per_method_top_token_plot(method_tokens: pd.DataFrame, path: Path, top_n: int = 12) -> Path | None:
    if method_tokens.empty:
        return None
    plot_frame = method_tokens.sort_values("abs_score", ascending=False).head(top_n).copy()
    plot_frame = plot_frame.iloc[::-1].copy()
    plot_frame["label"] = plot_frame.apply(
        lambda row: f"{short_token_label(str(row['token']), 22)} | {row['text_id']}",
        axis=1,
    )
    colors = np.where(plot_frame["score"].ge(0), "#2563eb", "#d97706")
    method_label = str(plot_frame.iloc[0]["metric_label"])
    y_pos = np.arange(len(plot_frame))
    fig, axis = plt.subplots(figsize=(11, max(4, len(plot_frame) * 0.43)))
    axis.barh(y_pos, plot_frame["abs_score"], color=colors, alpha=0.86)
    axis.set_yticks(y_pos)
    axis.set_yticklabels(plot_frame["label"], fontsize=8)
    axis.set_xlabel("Absolute score")
    axis.set_title(f"{method_label}: strongest token examples")
    axis.grid(axis="x", alpha=0.25, linestyle="--")
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


def save_html_report(
    html_report_path: Path,
    run_summary: pd.DataFrame,
    family_summary: pd.DataFrame,
    per_method_summary: pd.DataFrame,
    top_tokens: pd.DataFrame,
    qa_gaze_correlations: pd.DataFrame,
    qa_answer_synthesis: pd.DataFrame,
    plot_paths: Sequence[Path],
    per_method_plot_paths: Sequence[Path],
) -> None:
    exact_match_count = (
        int(qa_answer_synthesis["answer_exact_match"].sum())
        if "answer_exact_match" in qa_answer_synthesis
        else 0
    )
    qa_count = int(len(qa_answer_synthesis))
    mean_f1 = safe_float(
        pd.to_numeric(qa_answer_synthesis.get("answer_token_f1", pd.Series(dtype=float)), errors="coerce").mean()
    )
    strongest = strongest_correlation_rows(qa_gaze_correlations, 8)
    top_family = family_summary.sort_values("mean_abs", ascending=False).head(12)
    top_global_tokens = top_tokens.sort_values("abs_score", ascending=False).head(20)
    top_token_columns = ["family", "metric", "text_id", "language", "token", "score", "abs_score"]
    top_token_table = top_global_tokens.loc[:, top_token_columns].to_html(index=False, escape=True)
    family_names = ", ".join(sorted(map(str, run_summary["family"].dropna().unique()))) if not run_summary.empty else ""
    clean_method_columns = [
        "method_id",
        "metric_label",
        "model_name",
        "run_n",
        "text_n",
        "mean_abs_score",
        "strongest_text_id",
        "strongest_text_top_token",
        "strongest_token",
        "strongest_token_score",
        "strongest_gaze_metric",
        "strongest_gaze_mean_pearson_r",
    ]
    clean_methods = (
        per_method_summary.loc[:, [column for column in clean_method_columns if column in per_method_summary.columns]]
        if not per_method_summary.empty
        else pd.DataFrame()
    )

    plot_html = "\n".join(
        (
            f'<figure><img src="{relative_html_path(path, html_report_path.parent)}" '
            f'alt="{html.escape(path.stem)}"><figcaption>{html.escape(path.stem)}</figcaption></figure>'
        )
        for path in plot_paths
    )
    per_method_links = "\n".join(
        f'<li><a href="{relative_html_path(path, html_report_path.parent)}">{html.escape(path.stem)}</a></li>'
        for path in per_method_plot_paths
    )
    note_text = "This report aggregates already computed model artifacts."
    note_text += " Scores are summarized as signed values and absolute magnitudes."
    coverage_text = (
        f"Model-score runs: {len(run_summary)}. Model families: {html.escape(family_names)}. "
        f"QA exact matches: {exact_match_count}/{qa_count}. Mean QA token F1: {mean_f1:.3f}."
    )
    body = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Model Result Synthesis</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 32px; color: #1f2937; }}
    h1, h2 {{ color: #111827; }}
    .note {{ background: #f3f4f6; border-left: 4px solid #2563eb; padding: 12px 16px; }}
    figure {{ margin: 28px 0; }}
    img {{ max-width: 100%; border: 1px solid #e5e7eb; }}
    table {{ border-collapse: collapse; width: 100%; margin: 16px 0 28px; font-size: 13px; }}
    th, td {{ border: 1px solid #d1d5db; padding: 6px 8px; text-align: left; }}
    th {{ background: #f9fafb; }}
  </style>
</head>
<body>
  <h1>Model Result Synthesis</h1>
  <p class="note">{note_text}</p>
  <h2>Coverage</h2>
  <p>{coverage_text}</p>
  {plot_html}
  <h2>Clean Per-Method Summary</h2>
  {clean_methods.to_html(index=False, escape=True)}
  <h2>Per-Method Plots</h2>
  <ul>
    {per_method_links}
  </ul>
  <h2>Highest Mean Absolute Metrics</h2>
  {top_family.to_html(index=False, escape=True)}
  <h2>Strongest Mean QA-Gaze Correlations</h2>
  {strongest.to_html(index=False, escape=True)}
  <h2>Top Individual Tokens by Absolute Score</h2>
  {top_token_table}
</body>
</html>
"""
    html_report_path.write_text(body, encoding="utf-8")


def relative_html_path(path: Path, root: Path) -> str:
    return html.escape(str(path.relative_to(root)).replace("\\", "/"))


def strongest_correlation_rows(frame: pd.DataFrame, n: int) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    selected = frame.copy()
    selected["ranking_value"] = selected["mean_abs_pearson_r"].fillna(0.0)
    return selected.sort_values("ranking_value", ascending=False).head(n).drop(columns=["ranking_value"])


def method_identifier(family: str, metric: str) -> str:
    return safe_filename(f"{family}_{metric}")


def safe_filename(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    ascii_value = normalized.encode("ascii", "ignore").decode("ascii").lower()
    cleaned = re.sub(r"[^a-z0-9]+", "_", ascii_value).strip("_")
    return cleaned or "method"


def short_token_label(value: str, max_length: int) -> str:
    compact = re.sub(r"\s+", " ", value).strip()
    if len(compact) <= max_length:
        return compact
    return compact[: max(1, max_length - 3)] + ...


def strip_timestamp_prefix(value: str) -> str:
    return re.sub(r"^\d{14}_", "", value)


def slug_key(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    ascii_value = normalized.encode("ascii", "ignore").decode("ascii")
    lowered = ascii_value.lower()
    return re.sub(r"[^a-z0-9]+", "", lowered)


def normalize_token_for_overlap(value: str) -> str:
    return slug_key(value)


def overlap_score(candidate_key: str, manifest_key: str) -> int:
    if candidate_key in manifest_key or manifest_key in candidate_key:
        return min(len(candidate_key), len(manifest_key))
    max_score = 0
    for start in range(len(candidate_key)):
        for end in range(start + 8, len(candidate_key) + 1):
            fragment = candidate_key[start:end]
            if fragment in manifest_key:
                max_score = max(max_score, len(fragment))
    return max_score


def safe_float(value: object) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return math.nan
    return result
