import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

import matplotlib.patches as patches
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import numpy as np


@dataclass
class TokenRect:
    text: str
    x: float
    y: float
    width: float
    height: float

    def contains(self, px: float, py: float) -> bool:
        return (self.x <= px <= self.x + self.width) and (self.y <= py <= self.y + self.height)


@dataclass
class TokenGazeStats:
    token: str
    rect: TokenRect
    fixation_count: int = 0
    dwell_time: float = 0.0
    first_hit: float | None = None
    last_hit: float | None = None
    gaze_sample_count: int = 0
    visit_count: int = 0
    regression_in_count: int = 0
    regression_out_count: int = 0
    fixation_event_count: int = 0
    fixation_event_dwell_time: float = 0.0
    fixation_event_ids: list[int] = field(default_factory=list)
    visit_count_events: int = 0
    regression_in_count_events: int = 0
    regression_out_count_events: int = 0
    sample_indices: list[int] = field(default_factory=list)


def load_token_positions(path: Path) -> list[TokenRect]:
    if not path.exists():
        raise FileNotFoundError(f"Token positions file not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    rects: list[TokenRect] = []
    for entry in data:
        if not isinstance(entry, list) or len(entry) != 2:
            continue
        token, bbox = entry
        if not isinstance(bbox, dict):
            continue
        rects.append(
            TokenRect(
                text=str(token),
                x=float(bbox.get("x", 0.0)),
                y=float(bbox.get("y", 0.0)),
                width=float(bbox.get("width", 0.0)),
                height=float(bbox.get("height", 0.0)),
            )
        )
    return rects


def normalize_token_rects(tokens: list[TokenRect], screen_height: float) -> list[TokenRect]:
    return [
        TokenRect(
            text=rect.text,
            x=rect.x,
            y=screen_height - (rect.y + rect.height),
            width=rect.width,
            height=rect.height,
        )
        for rect in tokens
    ]


def normalize_word_boxes(words_data: Iterable[object], screen_height: float) -> list[tuple[str, dict[str, float]]]:
    normalized: list[tuple[str, dict[str, float]]] = []
    for entry in words_data:
        if not isinstance(entry, (list, tuple)) or len(entry) < 2:
            continue
        word, bbox = entry
        if not isinstance(bbox, dict):
            continue
        x = float(bbox.get("x", 0.0))
        y = float(bbox.get("y", 0.0))
        width = float(bbox.get("width", 0.0))
        height = float(bbox.get("height", 0.0))
        normalized.append(
            (
                str(word),
                {
                    "x": x,
                    "y": screen_height - (y + height),
                    "width": width,
                    "height": height,
                },
            )
        )
    return normalized


def apply_gaze_offset(
    xs: Iterable[float],
    ys: Iterable[float],
    offset_x: float = 0.0,
    offset_y: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    xs_arr = np.asarray(list(xs), dtype=float)
    ys_arr = np.asarray(list(ys), dtype=float)
    return xs_arr + float(offset_x), ys_arr + float(offset_y)


def _point_to_rect_distance(px: float, py: float, rect: TokenRect) -> float:
    dx = max(rect.x - px, 0.0, px - (rect.x + rect.width))
    dy = max(rect.y - py, 0.0, py - (rect.y + rect.height))
    return float(np.hypot(dx, dy))


def _point_to_rect_center_distance(px: float, py: float, rect: TokenRect) -> float:
    center_x = rect.x + rect.width * 0.5
    center_y = rect.y + rect.height * 0.5
    return float(np.hypot(px - center_x, py - center_y))


def _same_visual_line(py: float, rect: TokenRect, fixation_radius: float) -> bool:
    center_y = rect.y + rect.height * 0.5
    line_tolerance = max(rect.height, fixation_radius * 0.5)
    return abs(py - center_y) <= line_tolerance


def _map_fixation_to_token_index(
    cx: float,
    cy: float,
    tokens: list[TokenRect],
    fixation_radius: float,
) -> int | None:
    inside_candidates: list[tuple[float, float, int]] = []
    for idx, rect in enumerate(tokens):
        if rect.contains(cx, cy):
            inside_candidates.append(
                (_point_to_rect_center_distance(cx, cy, rect), rect.width * rect.height, idx)
            )
    if inside_candidates:
        inside_candidates.sort(key=lambda item: (item[0], item[1], item[2]))
        return inside_candidates[0][2]

    fallback_candidates: list[tuple[float, float, int]] = []
    for idx, rect in enumerate(tokens):
        if not _same_visual_line(cy, rect, fixation_radius):
            continue
        distance = _point_to_rect_distance(cx, cy, rect)
        if distance <= fixation_radius:
            fallback_candidates.append((distance, _point_to_rect_center_distance(cx, cy, rect), idx))
    if fallback_candidates:
        fallback_candidates.sort(key=lambda item: (item[0], item[1], item[2]))
        return fallback_candidates[0][2]

    return None


def _map_sample_to_token_index(
    px: float,
    py: float,
    tokens: list[TokenRect],
) -> int | None:
    inside_candidates: list[tuple[float, float, int]] = []
    for idx, rect in enumerate(tokens):
        if rect.contains(px, py):
            inside_candidates.append(
                (_point_to_rect_center_distance(px, py, rect), rect.width * rect.height, idx)
            )
    if not inside_candidates:
        return None
    inside_candidates.sort(key=lambda item: (item[0], item[1], item[2]))
    return inside_candidates[0][2]


def _aggregate_sentence_stats(stats: list[TokenGazeStats]) -> list[dict[str, object]]:
    if not stats:
        return []
    sentence_delimiters = {".", "!", "?"}
    sentences: list[dict[str, object]] = []
    current_indices: list[int] = []
    sentence_id = 1

    def _finalize_sentence(indices: list[int], sid: int) -> dict[str, object] | None:
        if not indices:
            return None
        sentence_tokens = [stats[i].token for i in indices]
        fixation_event_ids: set[int] = set()
        total_reading_time = 0.0
        visit_count_events = 0
        regressions_in_events = 0
        regressions_out_events = 0
        for i in indices:
            token_stat = stats[i]
            fixation_event_ids.update(token_stat.fixation_event_ids)
            total_reading_time += float(token_stat.fixation_event_dwell_time)
            visit_count_events += int(token_stat.visit_count_events)
            regressions_in_events += int(token_stat.regression_in_count_events)
            regressions_out_events += int(token_stat.regression_out_count_events)
        return {
            "sentence_id": sid,
            "sentence_text": " ".join(sentence_tokens),
            "token_indices": indices,
            "total_reading_time": total_reading_time,
            "fixation_event_count": len(fixation_event_ids),
            "visit_count_events": visit_count_events,
            "regressions_in_events": regressions_in_events,
            "regressions_out_events": regressions_out_events,
        }

    for idx, stat in enumerate(stats):
        current_indices.append(idx)
        if stat.token.strip() in sentence_delimiters:
            sentence = _finalize_sentence(current_indices, sentence_id)
            if sentence is not None:
                sentences.append(sentence)
                sentence_id += 1
            current_indices = []

    if current_indices:
        sentence = _finalize_sentence(current_indices, sentence_id)
        if sentence is not None:
            sentences.append(sentence)
    return sentences


def match_gaze_to_tokens(
    xs: Iterable[float],
    ys: Iterable[float],
    times: Iterable[float],
    tokens: list[TokenRect],
    fixation_events: list[dict[str, object]] | None = None,
    fixation_radius: float = 100.0,
) -> list[TokenGazeStats]:
    stats = [TokenGazeStats(token=rect.text, rect=rect) for rect in tokens]
    xs_arr = np.asarray(list(xs))
    ys_arr = np.asarray(list(ys))
    times_arr = np.asarray(list(times))
    if len(xs_arr) != len(ys_arr) or len(xs_arr) != len(times_arr):
        raise ValueError("Gaze coordinates and timestamps must be same length")
    active_token_index: int | None = None
    last_visited_token_index: int | None = None
    for idx, (px, py, t) in enumerate(zip(xs_arr, ys_arr, times_arr)):
        current_index = _map_sample_to_token_index(float(px), float(py), tokens)
        if current_index is None:
            active_token_index = None
            continue

        next_dt = float(times_arr[idx + 1] - times_arr[idx]) if idx + 1 < len(times_arr) else 0.0
        if next_dt < 0.0:
            next_dt = 0.0

        stat = stats[current_index]
        stat.fixation_count += 1
        stat.gaze_sample_count += 1
        stat.dwell_time += next_dt
        stat.sample_indices.append(idx)
        if stat.first_hit is None or t < stat.first_hit:
            stat.first_hit = float(t)
        if stat.last_hit is None or t > stat.last_hit:
            stat.last_hit = float(t)
        if active_token_index is None or current_index != active_token_index:
            stat.visit_count += 1
            if last_visited_token_index is not None and current_index < last_visited_token_index:
                stat.regression_in_count += 1
                stats[last_visited_token_index].regression_out_count += 1
            last_visited_token_index = current_index
        active_token_index = current_index

    if fixation_events:
        active_event_token_index: int | None = None
        last_visited_event_token_index: int | None = None
        for event_idx, event in enumerate(fixation_events):
            centroid = event.get("centroid")
            duration = float(event.get("duration", 0.0))
            if centroid is None or len(centroid) < 2:
                active_event_token_index = None
                continue
            cx = float(centroid[0])
            cy = float(centroid[1])
            current_index = _map_fixation_to_token_index(cx, cy, tokens, fixation_radius)
            if current_index is None:
                active_event_token_index = None
                continue
            stat = stats[current_index]
            stat.fixation_event_count += 1
            stat.fixation_event_dwell_time += duration
            stat.fixation_event_ids.append(int(event_idx))
            if active_event_token_index is None or current_index != active_event_token_index:
                stat.visit_count_events += 1
                if (
                    last_visited_event_token_index is not None
                    and current_index < last_visited_event_token_index
                ):
                    stat.regression_in_count_events += 1
                    stats[last_visited_event_token_index].regression_out_count_events += 1
                last_visited_event_token_index = current_index
            active_event_token_index = current_index

    return stats


def render_overlay(ax: Axes, stats: list[TokenGazeStats], cmap=plt.cm.viridis) -> None:
    if not stats:
        return
    counts = [
        stat.fixation_event_count if stat.fixation_event_count > 0 else stat.fixation_count
        for stat in stats
    ]
    max_count = max(counts, default=1)
    norm = plt.Normalize(vmin=0, vmax=max_count)
    for stat, count in zip(stats, counts):
        color = cmap(norm(count))
        rect = patches.Rectangle(
            (stat.rect.x, stat.rect.y),
            stat.rect.width,
            stat.rect.height,
            linewidth=1.5,
            edgecolor=color,
            facecolor=(color[0], color[1], color[2], 0.2),
        )
        ax.add_patch(rect)
        ax.text(
            stat.rect.x + stat.rect.width / 2,
            stat.rect.y + stat.rect.height / 2,
            stat.token,
            fontsize=7,
            ha="center",
            va="center",
            color="black",
        )


def serialize_token_stats(
    stats: list[TokenGazeStats],
    path: Path,
    metadata: dict[str, str] | None = None,
    sentences: list[dict[str, object]] | None = None,
) -> None:
    if sentences is None:
        sentences = _aggregate_sentence_stats(stats)
    payload = {
        "metadata": metadata or {},
        "tokens": [
            {
                "token": stat.token,
                "fixation_count": stat.fixation_count,
                "dwell_time": stat.dwell_time,
                "total_gaze": stat.dwell_time,
                "first_gaze": stat.first_hit,
                "last_gaze": stat.last_hit,
                "gaze_sample_count": stat.gaze_sample_count,
                "visit_count": stat.visit_count,
                "regressions_in": stat.regression_in_count,
                "regressions_out": stat.regression_out_count,
                "fixation_event_count": stat.fixation_event_count,
                "fixation_event_dwell_time": stat.fixation_event_dwell_time,
                "fixation_event_ids": stat.fixation_event_ids,
                "visit_count_events": stat.visit_count_events,
                "regressions_in_events": stat.regression_in_count_events,
                "regressions_out_events": stat.regression_out_count_events,
                "rect": asdict(stat.rect),
                "first_hit": stat.first_hit,
                "last_hit": stat.last_hit,
            }
            for stat in stats
        ],
        "sentences": sentences,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_token_dashboard(
    stats: list[TokenGazeStats],
    points: Iterable[tuple[float, float]],
    output: Path,
    metadata: dict[str, str] | None = None,
    sentences: list[dict[str, object]] | None = None,
) -> None:
    def _fmt(value: float | None) -> str:
        return f"{value:.2f}" if value is not None else "-"

    table_rows = "\n".join(
        "<tr>"
        f"<td>{stat.token}</td>"
        f"<td>{stat.fixation_event_count}</td>"
        f"<td>{stat.fixation_count}</td>"
        f"<td>{stat.fixation_event_dwell_time:.2f}</td>"
        f"<td>{stat.dwell_time:.2f}</td>"
        f"<td>{_fmt(stat.first_hit)}</td>"
        f"<td>{_fmt(stat.last_hit)}</td>"
        f"<td>{stat.visit_count_events}</td>"
        f"<td>{stat.visit_count}</td>"
        f"<td>{stat.regression_in_count_events}</td>"
        f"<td>{stat.regression_in_count}</td>"
        "</tr>"
        for stat in stats
    )
    if sentences is None:
        sentences = _aggregate_sentence_stats(stats)
    sentence_rows = "\n".join(
        "<tr>"
        f"<td>{sentence['sentence_id']}</td>"
        f"<td>{sentence['sentence_text']}</td>"
        f"<td>{sentence['fixation_event_count']}</td>"
        f"<td>{sentence['total_reading_time']:.2f}</td>"
        f"<td>{sentence['visit_count_events']}</td>"
        f"<td>{sentence['regressions_in_events']}</td>"
        "</tr>"
        for sentence in sentences
    )
    rows_html = f"""
    <table border="1" cellpadding="4" cellspacing="0">
        <tr>
            <th>Token</th>
            <th>Fixation events (robust)</th>
            <th>Fixation count (legacy/sample-based)</th>
            <th>Total gaze (s, robust event-based)</th>
            <th>Total gaze (s, legacy/sample-based)</th>
            <th>First gaze (s)</th>
            <th>Last gaze (s)</th>
            <th>Visits (robust)</th>
            <th>Visits (legacy)</th>
            <th>Regressions in (robust)</th>
            <th>Regressions in (legacy)</th>
        </tr>
        {table_rows}
    </table>
    """
    sentences_html = f"""
    <h2>Sentence Reading Stats (robust event-based)</h2>
    <table border="1" cellpadding="4" cellspacing="0">
        <tr>
            <th>Sentence ID</th>
            <th>Sentence text</th>
            <th>Fixation events</th>
            <th>Total reading time (s)</th>
            <th>Visits</th>
            <th>Regressions in</th>
        </tr>
        {sentence_rows}
    </table>
    """
    list_points = "".join(f"<li>{float(x):.1f}, {float(y):.1f}</li>" for x, y in points)
    html = f"""
    <html>
    <head><title>Token Gaze Dashboard</title></head>
    <body>
        <h1>Token Gaze Stats</h1>
        <p>Metadata: {json.dumps(metadata or {}, ensure_ascii=False)}</p>
        {rows_html}
        {sentences_html}
        <h2>Gaze path (raw points)</h2>
        <ol>{list_points}</ol>
    </body>
    </html>
    """
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html, encoding="utf-8")
