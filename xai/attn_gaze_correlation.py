import math
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from statsmodels.stats.multitest import multipletests

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_GAZE_METRICS: tuple[str, ...] = (
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

DEFAULT_SKIP_TOKENS: frozenset[str] = frozenset({".", ","})



@dataclass(frozen=True)
class CorrelationConfig:
    gaze_csv_path: Path
    attention_csv_path: Path
    output_dir: Path
    gaze_metrics: tuple[str, ...] = DEFAULT_GAZE_METRICS
    token_column: str = "token"
    skip_tokens: frozenset[str] = DEFAULT_SKIP_TOKENS
    min_observations: int = 3
    top_plot_count: int = 6


@dataclass(frozen=True)
class CorrelationRunResult:
    aligned_tokens: pd.DataFrame
    correlations: pd.DataFrame
    plot_manifest: pd.DataFrame
    aligned_tokens_path: Path
    correlations_path: Path
    plot_manifest_path: Path


@dataclass(frozen=True)
class MetricPairStatistics:
    gaze_metric: str
    attention_metric: str
    n: int
    pearson_r: float
    pearson_p: float
    pearson_ci_low: float
    pearson_ci_high: float
    spearman_rho: float
    spearman_p: float
    kendall_tau: float
    kendall_p: float
    linear_slope: float
    linear_intercept: float
    linear_r_squared: float
    status: str


class AttentionGazeCorrelationBenchmark:
    def __init__(self, config: CorrelationConfig) -> None:
        self._config = config

    def run(self) -> CorrelationRunResult:
        self._config.output_dir.mkdir(parents=True, exist_ok=True)
        gaze_frame = self._load_gaze_frame()
        attention_frame = self._load_attention_frame()
        aligned_tokens = self._align_tokens(gaze_frame, attention_frame)
        correlations = self._compute_correlations(aligned_tokens)
        correlations = self._add_fdr_columns(correlations)
        plot_manifest = self._save_plots(aligned_tokens, correlations)

        aligned_tokens_path = self._config.output_dir / "aligned_tokens.csv"
        correlations_path = self._config.output_dir / "correlation_statistics.csv"
        plot_manifest_path = self._config.output_dir / "plot_manifest.csv"

        aligned_tokens.to_csv(aligned_tokens_path, index=False)
        correlations.to_csv(correlations_path, index=False)
        plot_manifest.to_csv(plot_manifest_path, index=False)

        return CorrelationRunResult(
            aligned_tokens=aligned_tokens,
            correlations=correlations,
            plot_manifest=plot_manifest,
            aligned_tokens_path=aligned_tokens_path,
            correlations_path=correlations_path,
            plot_manifest_path=plot_manifest_path,
        )

    def _load_gaze_frame(self) -> pd.DataFrame:
        frame = pd.read_csv(self._config.gaze_csv_path)
        required_columns = {self._config.token_column, *self._config.gaze_metrics}
        missing_columns = required_columns.difference(frame.columns)
        if missing_columns:
            raise ValueError(f"Gaze CSV missing required columns: {sorted(missing_columns)}")

        selected_columns = [self._config.token_column, *self._config.gaze_metrics]
        if "token_index" in frame.columns:
            selected_columns.insert(0, "token_index")

        selected = frame.loc[:, selected_columns].copy()
        selected = selected[~selected[self._config.token_column].isin(self._config.skip_tokens)].copy()
        return self._add_alignment_keys(selected, source_prefix="gaze")

    def _load_attention_frame(self) -> pd.DataFrame:
        frame = pd.read_csv(self._config.attention_csv_path, sep=None, engine="python")
        if self._config.token_column not in frame.columns:
            raise ValueError(f"Attention CSV missing token column: {self._config.token_column}")

        numeric_columns = [
            column
            for column in frame.columns
            if column != self._config.token_column and pd.api.types.is_numeric_dtype(frame[column])
        ]
        if not numeric_columns:
            raise ValueError("Attention CSV has no numeric attention metric columns.")

        selected = frame.loc[:, [self._config.token_column, *numeric_columns]].copy()
        selected = selected[~selected[self._config.token_column].isin(self._config.skip_tokens)].copy()
        return self._add_alignment_keys(selected, source_prefix="attention")

    def _add_alignment_keys(self, frame: pd.DataFrame, source_prefix: str) -> pd.DataFrame:
        prepared = frame.copy()
        token_column = self._config.token_column
        prepared[f"{source_prefix}_token"] = prepared[token_column].astype(str)
        prepared["normalized_token"] = prepared[f"{source_prefix}_token"].map(normalize_token_for_alignment)
        prepared = prepared[prepared["normalized_token"] != ""].copy()
        prepared["token_occurrence"] = prepared.groupby("normalized_token", sort=False).cumcount()
        prepared[f"{source_prefix}_sequence_index"] = np.arange(len(prepared), dtype=int)
        return prepared.drop(columns=[token_column])

    def _align_tokens(self, gaze_frame: pd.DataFrame, attention_frame: pd.DataFrame) -> pd.DataFrame:
        aligned = pd.merge(
            gaze_frame,
            attention_frame,
            on=["normalized_token", "token_occurrence"],
            how="inner",
            validate="one_to_one",
            suffixes=("_gaze", "_attention"),
        )
        if aligned.empty:
            raise ValueError("No tokens aligned between gaze and attention CSVs.")

        aligned = aligned.sort_values(["attention_sequence_index", "gaze_sequence_index"]).reset_index(drop=True)
        aligned.insert(0, "aligned_index", np.arange(len(aligned), dtype=int))
        return aligned

    def _compute_correlations(self, aligned_tokens: pd.DataFrame) -> pd.DataFrame:
        attention_metrics = self._attention_metric_columns(aligned_tokens)
        rows: list[dict[str, float | int | str]] = []

        for gaze_metric in self._config.gaze_metrics:
            for attention_metric in attention_metrics:
                pair_frame = aligned_tokens.loc[:, [gaze_metric, attention_metric]]
                pair_frame = pair_frame.apply(pd.to_numeric, errors="coerce")
                pair_frame = pair_frame.replace([np.inf, -np.inf], np.nan).dropna()
                statistics = self._compute_metric_pair_statistics(
                    gaze_metric=gaze_metric,
                    attention_metric=attention_metric,
                    pair_frame=pair_frame,
                )
                rows.append(dataclass_to_dict(statistics))

        return pd.DataFrame(rows)

    def _compute_metric_pair_statistics(
        self,
        gaze_metric: str,
        attention_metric: str,
        pair_frame: pd.DataFrame,
    ) -> MetricPairStatistics:
        n = int(len(pair_frame))
        empty_result = MetricPairStatistics(
            gaze_metric=gaze_metric,
            attention_metric=attention_metric,
            n=n,
            pearson_r=math.nan,
            pearson_p=math.nan,
            pearson_ci_low=math.nan,
            pearson_ci_high=math.nan,
            spearman_rho=math.nan,
            spearman_p=math.nan,
            kendall_tau=math.nan,
            kendall_p=math.nan,
            linear_slope=math.nan,
            linear_intercept=math.nan,
            linear_r_squared=math.nan,
            status="not_computed",
        )
        if n < self._config.min_observations:
            return replace_status(empty_result, f"too_few_observations_min_{self._config.min_observations}")

        x_values = pair_frame[attention_metric].to_numpy(dtype=float)
        y_values = pair_frame[gaze_metric].to_numpy(dtype=float)
        if is_constant(x_values) or is_constant(y_values):
            return replace_status(empty_result, "constant_input")

        pearson_result = stats.pearsonr(x_values, y_values)
        spearman_result = stats.spearmanr(x_values, y_values, nan_policy="omit")
        kendall_result = stats.kendalltau(x_values, y_values, nan_policy="omit")
        linregress_result = stats.linregress(x_values, y_values)
        pearson_ci = pearson_result.confidence_interval(confidence_level=0.95)

        return MetricPairStatistics(
            gaze_metric=gaze_metric,
            attention_metric=attention_metric,
            n=n,
            pearson_r=float(pearson_result.statistic),
            pearson_p=float(pearson_result.pvalue),
            pearson_ci_low=float(pearson_ci.low),
            pearson_ci_high=float(pearson_ci.high),
            spearman_rho=float(spearman_result.statistic),
            spearman_p=float(spearman_result.pvalue),
            kendall_tau=float(kendall_result.statistic),
            kendall_p=float(kendall_result.pvalue),
            linear_slope=float(linregress_result.slope),
            linear_intercept=float(linregress_result.intercept),
            linear_r_squared=float(linregress_result.rvalue**2),
            status="ok",
        )

    def _add_fdr_columns(self, correlations: pd.DataFrame) -> pd.DataFrame:
        adjusted = correlations.copy()
        for p_column, q_column in (
            ("pearson_p", "pearson_fdr_q"),
            ("spearman_p", "spearman_fdr_q"),
            ("kendall_p", "kendall_fdr_q"),
        ):
            adjusted[q_column] = math.nan
            valid_mask = adjusted[p_column].notna()
            if valid_mask.any():
                adjusted.loc[valid_mask, q_column] = multipletests(
                    adjusted.loc[valid_mask, p_column].astype(float),
                    method="fdr_bh",
                )[1]
        return adjusted

    def _save_plots(self, aligned_tokens: pd.DataFrame, correlations: pd.DataFrame) -> pd.DataFrame:
        plots_dir = self._config.output_dir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        rows: list[dict[str, str]] = []

        rows.extend(self._save_heatmap_plots(correlations, plots_dir))
        rows.extend(self._save_top_scatter_plots(aligned_tokens, correlations, plots_dir))
        profile_path = plots_dir / "token_metric_profiles.png"
        self._save_token_profile_plot(aligned_tokens, profile_path)
        rows.append({"plot_type": "token_profile", "path": str(profile_path)})
        return pd.DataFrame(rows)

    def _save_heatmap_plots(self, correlations: pd.DataFrame, plots_dir: Path) -> list[dict[str, str]]:
        plot_rows: list[dict[str, str]] = []
        for value_column, plot_type, title in (
            ("pearson_r", "pearson_heatmap", "Pearson r: attention metrics vs gaze metrics"),
            ("spearman_rho", "spearman_heatmap", "Spearman rho: attention metrics vs gaze metrics"),
        ):
            pivot = correlations.pivot(index="gaze_metric", columns="attention_metric", values=value_column)
            plot_path = plots_dir / f"{plot_type}.png"
            fig_width = max(10.0, 0.75 * len(pivot.columns))
            fig_height = max(6.0, 0.45 * len(pivot.index))
            fig, axis = plt.subplots(figsize=(fig_width, fig_height))
            sns.heatmap(
                pivot,
                annot=True,
                fmt=".2f",
                cmap="coolwarm",
                center=0.0,
                vmin=-1.0,
                vmax=1.0,
                linewidths=0.4,
                cbar_kws={"label": value_column},
                ax=axis,
            )
            axis.set_title(title)
            axis.set_xlabel("Attention metric")
            axis.set_ylabel("Gaze metric")
            fig.tight_layout()
            fig.savefig(plot_path, dpi=220, bbox_inches="tight")
            plt.close(fig)
            plot_rows.append({"plot_type": plot_type, "path": str(plot_path)})
        return plot_rows

    def _save_top_scatter_plots(
        self,
        aligned_tokens: pd.DataFrame,
        correlations: pd.DataFrame,
        plots_dir: Path,
    ) -> list[dict[str, str]]:
        ok_correlations = correlations[correlations["status"] == "ok"].copy()
        if ok_correlations.empty:
            return []

        ok_correlations["absolute_pearson_r"] = ok_correlations["pearson_r"].abs()
        top_rows = ok_correlations.sort_values("absolute_pearson_r", ascending=False).head(self._config.top_plot_count)
        column_count = min(3, len(top_rows))
        row_count = math.ceil(len(top_rows) / column_count)
        fig, axes = plt.subplots(row_count, column_count, figsize=(5.8 * column_count, 4.8 * row_count))
        flattened_axes = np.asarray(axes, dtype=object).reshape(-1)

        for axis, (_, row) in zip(flattened_axes, top_rows.iterrows()):
            attention_metric = str(row["attention_metric"])
            gaze_metric = str(row["gaze_metric"])
            plot_frame = aligned_tokens[[attention_metric, gaze_metric, "attention_token"]].copy()
            sns.regplot(
                data=plot_frame,
                x=attention_metric,
                y=gaze_metric,
                scatter_kws={"s": 38, "alpha": 0.78},
                line_kws={"color": "#dc2626", "linewidth": 1.4},
                ax=axis,
            )
            title = f"{attention_metric} vs {gaze_metric}\nr={row['pearson_r']:.3f}, q={row['pearson_fdr_q']:.3g}"
            axis.set_title(title)
            for _, token_row in plot_frame.iterrows():
                axis.annotate(
                    str(token_row["attention_token"]),
                    (float(token_row[attention_metric]), float(token_row[gaze_metric])),
                    fontsize=7,
                    alpha=0.72,
                    xytext=(3, 3),
                    textcoords="offset points",
                )

        for axis in flattened_axes[len(top_rows) :]:
            axis.set_visible(False)

        fig.tight_layout()
        plot_path = plots_dir / "top_attention_gaze_scatterplots.png"
        fig.savefig(plot_path, dpi=220, bbox_inches="tight")
        plt.close(fig)
        return [{"plot_type": "top_scatterplots", "path": str(plot_path)}]

    def _save_token_profile_plot(self, aligned_tokens: pd.DataFrame, plot_path: Path) -> None:
        profile_metrics = [
            metric
            for metric in (
                "fixation_event_dwell_time_mean",
                "dwell_time_mean",
                "fixation_event_count_mean",
                "normentw",
                "deltaentw",
                "manhattanw",
                "emdw",
                "surprisal",
                "nli",
                "masking",
            )
            if metric in aligned_tokens.columns
        ]
        if not profile_metrics:
            return

        profile_frame = aligned_tokens[["aligned_index", "attention_token", *profile_metrics]].copy()
        for metric in profile_metrics:
            values = profile_frame[metric].astype(float)
            value_range = float(values.max() - values.min())
            if value_range == 0.0:
                profile_frame[f"{metric}_scaled"] = 0.0
            else:
                profile_frame[f"{metric}_scaled"] = (values - float(values.min())) / value_range

        long_frame = profile_frame.melt(
            id_vars=["aligned_index", "attention_token"],
            value_vars=[f"{metric}_scaled" for metric in profile_metrics],
            var_name="metric",
            value_name="scaled_value",
        )
        long_frame["metric"] = long_frame["metric"].str.removesuffix("_scaled")

        fig, axis = plt.subplots(figsize=(13.0, 6.2))
        sns.lineplot(data=long_frame, x="aligned_index", y="scaled_value", hue="metric", marker="o", ax=axis)
        axis.set_title("Scaled token-level attention and gaze profiles")
        axis.set_xlabel("Aligned token index")
        axis.set_ylabel("Min-max scaled value")
        axis.set_xticks(profile_frame["aligned_index"])
        axis.set_xticklabels(profile_frame["attention_token"], rotation=70, ha="right", fontsize=8)
        axis.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), title="Metric")
        fig.tight_layout()
        fig.savefig(plot_path, dpi=220, bbox_inches="tight")
        plt.close(fig)

    def _attention_metric_columns(self, aligned_tokens: pd.DataFrame) -> list[str]:
        excluded_columns = {
            "aligned_index",
            "token_index",
            "gaze_token",
            "attention_token",
            "normalized_token",
            "token_occurrence",
            "gaze_sequence_index",
            "attention_sequence_index",
            *self._config.gaze_metrics,
        }
        return [
            column
            for column in aligned_tokens.columns
            if column not in excluded_columns and pd.api.types.is_numeric_dtype(aligned_tokens[column])
        ]


def normalize_token_for_alignment(token: str) -> str:
    normalized = re.sub(r"\s+", " ", str(token).strip()).casefold()
    return "".join(char for char in normalized if not unicodedata.category(char).startswith("P"))


def is_constant(values: np.ndarray) -> bool:
    finite_values = values[np.isfinite(values)]
    return len(finite_values) == 0 or bool(np.allclose(finite_values, finite_values[0]))


def replace_status(statistics: MetricPairStatistics, status: str) -> MetricPairStatistics:
    return MetricPairStatistics(
        gaze_metric=statistics.gaze_metric,
        attention_metric=statistics.attention_metric,
        n=statistics.n,
        pearson_r=statistics.pearson_r,
        pearson_p=statistics.pearson_p,
        pearson_ci_low=statistics.pearson_ci_low,
        pearson_ci_high=statistics.pearson_ci_high,
        spearman_rho=statistics.spearman_rho,
        spearman_p=statistics.spearman_p,
        kendall_tau=statistics.kendall_tau,
        kendall_p=statistics.kendall_p,
        linear_slope=statistics.linear_slope,
        linear_intercept=statistics.linear_intercept,
        linear_r_squared=statistics.linear_r_squared,
        status=status,
    )


def dataclass_to_dict(statistics: MetricPairStatistics) -> dict[str, float | int | str]:
    return {
        "gaze_metric": statistics.gaze_metric,
        "attention_metric": statistics.attention_metric,
        "n": statistics.n,
        "pearson_r": statistics.pearson_r,
        "pearson_p": statistics.pearson_p,
        "pearson_ci_low": statistics.pearson_ci_low,
        "pearson_ci_high": statistics.pearson_ci_high,
        "spearman_rho": statistics.spearman_rho,
        "spearman_p": statistics.spearman_p,
        "kendall_tau": statistics.kendall_tau,
        "kendall_p": statistics.kendall_p,
        "linear_slope": statistics.linear_slope,
        "linear_intercept": statistics.linear_intercept,
        "linear_r_squared": statistics.linear_r_squared,
        "status": statistics.status,
    }
