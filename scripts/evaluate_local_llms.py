"""Run and render the production-derived local-LLM ad-detection benchmark.

The input corpus is intentionally external to the repository because it contains
production transcripts. The committed JSON and HTML outputs contain only episode
IDs, timing intervals, aggregate metrics, and safe provider diagnostics.
"""

from __future__ import annotations

import argparse
import html
import json
import logging
import math
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.core.ai_services import AdDetector  # noqa: E402
from app.core.processor import Processor  # noqa: E402


DEFAULT_OPTIONS = {
    "remove_ads": True,
    "remove_promos": True,
    "remove_intros": False,
    "remove_outros": False,
    "custom_instructions": None,
}
SECRET_PATTERN = re.compile(
    r"(?i)(?:sk-or-v1-|sk-|key-)[a-z0-9_-]{12,}|bearer\s+[a-z0-9._-]+"
)


class EvaluationAdDetector(AdDetector):
    """AdDetector using isolated in-memory settings instead of the application DB."""

    def __init__(self, evaluation_settings: dict[str, Any]):
        self._evaluation_settings = dict(evaluation_settings)
        super().__init__()

    def _load_settings(self) -> dict[str, Any]:
        return dict(self._evaluation_settings)


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def load_env_value(path: Path, name: str) -> str | None:
    value = os.environ.get(name)
    if value:
        return value
    if not path.exists():
        return None
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, candidate = line.split("=", 1)
        if key.strip() == name:
            return candidate.strip().strip("\"'")
    return None


def normalized_intervals(segments: list[dict[str, Any]]) -> list[tuple[float, float]]:
    intervals: list[tuple[float, float]] = []
    for segment in segments:
        try:
            start = max(0.0, float(segment["start"]))
            end = float(segment["end"])
        except (KeyError, TypeError, ValueError):
            continue
        if not math.isfinite(start) or not math.isfinite(end) or end <= start:
            continue
        if intervals and start <= intervals[-1][1]:
            intervals[-1] = (intervals[-1][0], max(intervals[-1][1], end))
        else:
            intervals.append((start, end))
    return intervals


def interval_duration(intervals: list[tuple[float, float]]) -> float:
    return sum(end - start for start, end in intervals)


def interval_intersection(
    left: list[tuple[float, float]], right: list[tuple[float, float]]
) -> float:
    overlap = 0.0
    left_index = right_index = 0
    while left_index < len(left) and right_index < len(right):
        left_start, left_end = left[left_index]
        right_start, right_end = right[right_index]
        overlap += max(0.0, min(left_end, right_end) - max(left_start, right_start))
        if left_end <= right_end:
            left_index += 1
        else:
            right_index += 1
    return overlap


def interval_metrics(
    predicted: list[dict[str, Any]], reference: list[dict[str, Any]]
) -> dict[str, float]:
    predicted_intervals = normalized_intervals(
        sorted(predicted, key=lambda item: float(item.get("start", 0)))
    )
    reference_intervals = normalized_intervals(
        sorted(reference, key=lambda item: float(item.get("start", 0)))
    )
    predicted_seconds = interval_duration(predicted_intervals)
    reference_seconds = interval_duration(reference_intervals)
    overlap_seconds = interval_intersection(predicted_intervals, reference_intervals)
    precision = overlap_seconds / predicted_seconds if predicted_seconds else 0.0
    recall = overlap_seconds / reference_seconds if reference_seconds else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "predicted_seconds": round(predicted_seconds, 2),
        "reference_seconds": round(reference_seconds, 2),
        "overlap_seconds": round(overlap_seconds, 2),
    }


def segment_match(
    segment: dict[str, Any], other_segments: list[dict[str, Any]]
) -> dict[str, Any]:
    start = float(segment["start"])
    end = float(segment["end"])
    duration = end - start
    best_index = None
    best_overlap = 0.0
    for index, other in enumerate(other_segments):
        overlap = max(
            0.0,
            min(end, float(other["end"])) - max(start, float(other["start"])),
        )
        if overlap > best_overlap:
            best_overlap = overlap
            best_index = index
    return {
        "best_reference_index": best_index,
        "overlap_seconds": round(best_overlap, 2),
        "overlap_fraction": round(best_overlap / duration, 4) if duration > 0 else 0.0,
        "match": "matched" if best_overlap > 0 else "extra",
    }


def safe_segment(
    segment: dict[str, Any], references: list[dict[str, Any]]
) -> dict[str, Any]:
    start = round(float(segment["start"]), 2)
    end = round(float(segment["end"]), 2)
    safe = {
        "start": start,
        "end": end,
        "duration": round(end - start, 2),
        "label": str(segment.get("label") or "Non-Content")[:80],
    }
    safe.update(segment_match(safe, references))
    return safe


def safe_error(exc: Exception) -> dict[str, str]:
    message = SECRET_PATTERN.sub("[redacted]", str(exc))
    message = " ".join(message.split())
    return {
        "type": type(exc).__name__,
        "message": message[:500],
    }


def numeric_usage(metadata: dict[str, Any]) -> dict[str, float | int]:
    usage = {}
    for key, value in (metadata.get("usage") or {}).items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        usage[str(key)] = round(value, 8) if isinstance(value, float) else value
    return usage


def usage_tokens(usage: dict[str, Any], kind: str) -> int:
    candidates = {
        "prompt": ("prompt_tokens", "input_tokens"),
        "completion": ("completion_tokens", "output_tokens"),
    }[kind]
    for key in candidates:
        value = usage.get(key)
        if isinstance(value, (int, float)):
            return int(value)
    return 0


def estimated_cost(model: dict[str, Any], usage: dict[str, Any]) -> float:
    return round(
        usage_tokens(usage, "prompt") * float(model["prompt_price_per_token"])
        + usage_tokens(usage, "completion")
        * float(model["completion_price_per_token"]),
        6,
    )


def build_settings(model: dict[str, Any], api_key: str) -> dict[str, Any]:
    return {
        "active_ai_provider": "openrouter",
        "openrouter_api_key": api_key,
        "openrouter_model": json.dumps([model["model_id"]]),
        "ad_chunking_enabled": int(bool(model["chunking_enabled"])),
        "ad_chunk_context_tokens": model.get("declared_context_tokens") or 8192,
        "ad_chunk_overlap_seconds": 30,
        "ad_chunk_max_chunks": 32,
        "ad_include_reasons": 0,
        "whitelist_mode": 1,
    }


def safe_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "provider",
        "model",
        "chunking_enabled",
        "chunk_count",
        "completed_chunks",
        "estimated_input_tokens",
        "context_window_tokens",
        "elapsed_seconds",
        "finish_reasons",
        "complete",
        "failed_chunk",
        "error_type",
    )
    result = {key: metadata[key] for key in keys if key in metadata}
    result["usage"] = numeric_usage(metadata)
    return result


def evaluate_episode(
    detector: EvaluationAdDetector,
    model: dict[str, Any],
    episode: dict[str, Any],
    quality_threshold: float,
) -> dict[str, Any]:
    references = [
        {
            "start": round(float(segment["start"]), 2),
            "end": round(float(segment["end"]), 2),
            "label": str(segment.get("label") or "Historical removal")[:80],
        }
        for segment in episode["reference_segments"]
    ]
    started = time.monotonic()
    try:
        raw_segments = detector.detect_ads(
            episode["transcript"],
            options=DEFAULT_OPTIONS,
            whitelist_mode=True,
        )
        predicted = Processor._prepare_remove_segments(
            raw_segments,
            whitelist_mode=True,
            total_duration=float(episode["duration"]),
        )
        metrics = interval_metrics(predicted, references)
        metadata = safe_metadata(detector.last_detection_metadata)
        execution_pass = bool(metadata.get("complete"))
        quality_pass = execution_pass and metrics["f1"] >= quality_threshold
        return {
            "episode_id": episode["id"],
            "execution_status": "pass" if execution_pass else "fail",
            "quality_status": "pass" if quality_pass else "fail",
            "metrics": metrics,
            "predicted_segments": [
                safe_segment(segment, references) for segment in predicted
            ],
            "metadata": metadata,
            "estimated_cost_usd": estimated_cost(model, metadata["usage"]),
            "wall_seconds": round(time.monotonic() - started, 3),
        }
    except Exception as exc:
        metadata = safe_metadata(detector.last_detection_metadata)
        return {
            "episode_id": episode["id"],
            "execution_status": "fail",
            "quality_status": "fail",
            "metrics": None,
            "predicted_segments": [],
            "metadata": metadata,
            "estimated_cost_usd": estimated_cost(model, metadata["usage"]),
            "wall_seconds": round(time.monotonic() - started, 3),
            "error": safe_error(exc),
        }


def summarize_run(run: dict[str, Any]) -> dict[str, Any]:
    episodes = run["episodes"]
    completed = [item for item in episodes if item["execution_status"] == "pass"]
    scored = [item for item in completed if item["metrics"] is not None]
    total_cost = sum(float(item["estimated_cost_usd"]) for item in episodes)
    total_seconds = sum(float(item["wall_seconds"]) for item in episodes)
    average_f1 = (
        sum(float(item["metrics"]["f1"]) for item in scored) / len(scored)
        if scored
        else 0.0
    )
    return {
        "execution_passes": len(completed),
        "execution_total": len(episodes),
        "quality_passes": sum(
            item["quality_status"] == "pass" for item in episodes
        ),
        "average_f1": round(average_f1, 4),
        "estimated_cost_usd": round(total_cost, 6),
        "wall_seconds": round(total_seconds, 3),
        "overall_status": (
            "pass"
            if episodes
            and all(item["execution_status"] == "pass" for item in episodes)
            and all(item["quality_status"] == "pass" for item in episodes)
            else "fail"
        ),
    }


def safe_corpus_summary(corpus: dict[str, Any]) -> list[dict[str, Any]]:
    summaries = []
    for episode in corpus["episodes"]:
        transcript_segments = episode["transcript"].get("segments") or []
        transcript_characters = sum(
            len(str(segment.get("text") or "")) for segment in transcript_segments
        )
        references = [
            {
                "start": round(float(segment["start"]), 2),
                "end": round(float(segment["end"]), 2),
                "duration": round(
                    float(segment["end"]) - float(segment["start"]), 2
                ),
                "label": str(segment.get("label") or "Historical removal")[:80],
            }
            for segment in episode["reference_segments"]
        ]
        summaries.append(
            {
                "episode_id": episode["id"],
                "case": episode["case"],
                "duration": round(float(episode["duration"]), 2),
                "transcript_segments": len(transcript_segments),
                "transcript_characters": transcript_characters,
                "reference_segments": references,
                "reference_seconds": round(
                    sum(segment["duration"] for segment in references), 2
                ),
            }
        )
    return summaries


def build_results(
    matrix: dict[str, Any],
    corpus: dict[str, Any],
    existing: dict[str, Any] | None = None,
) -> dict[str, Any]:
    results = existing or {}
    results.update(
        {
            "schema_version": 1,
            "updated_at": utc_now(),
            "catalog_checked_at": matrix["catalog_checked_at"],
            "provider": matrix["provider"],
            "base_url": matrix["base_url"],
            "quality_f1_threshold": float(matrix["quality_f1_threshold"]),
            "test_configuration": {
                "mode": "Whitelist ad detection",
                "chunk_overlap_seconds": 30,
                "chunk_max_count": 32,
                "reasons_enabled": False,
                "temperature": (
                    "0 for chunked runs; provider default for standard control"
                ),
                "reference": corpus.get(
                    "reference_kind", "Historical production remove windows"
                ),
                "production_access": "Read-only",
            },
            "episodes": safe_corpus_summary(corpus),
        }
    )
    results.setdefault("model_runs", [])
    return results


def h(value: Any) -> str:
    return html.escape(str(value), quote=True)


def clock(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return (
        f"{hours}:{minutes:02d}:{seconds:02d}"
        if hours
        else f"{minutes}:{seconds:02d}"
    )


def status_badge(status: str, label: str | None = None) -> str:
    safe_status = "pass" if status == "pass" else "fail"
    return f'<span class="badge {safe_status}">{h(label or safe_status.upper())}</span>'


def segment_list(segments: list[dict[str, Any]]) -> str:
    if not segments:
        return '<span class="muted">No removal windows detected</span>'
    rows = []
    for segment in segments:
        match = segment.get("match")
        match_text = (
            f"matches reference {int(segment['best_reference_index']) + 1}, "
            f"{segment['overlap_seconds']:.2f}s overlap"
            if match == "matched"
            else "no reference overlap"
        )
        rows.append(
            "<li>"
            f"<strong>{clock(segment['start'])}–{clock(segment['end'])}</strong> "
            f"<span>{segment['duration']:.2f}s · {h(segment['label'])}</span>"
            f'<small class="{h(match or "extra")}">{h(match_text)}</small>'
            "</li>"
        )
    return f'<ol class="segments">{"".join(rows)}</ol>'


def timeline(
    duration: float,
    references: list[dict[str, Any]],
    predicted: list[dict[str, Any]],
) -> str:
    def bars(items: list[dict[str, Any]], class_name: str) -> str:
        parts = []
        for item in items:
            left = max(0.0, min(100.0, float(item["start"]) / duration * 100))
            width = max(
                0.25,
                min(
                    100.0 - left,
                    (float(item["end"]) - float(item["start"])) / duration * 100,
                ),
            )
            parts.append(
                f'<i class="{class_name}" style="left:{left:.4f}%;width:{width:.4f}%" '
                f'title="{clock(float(item["start"]))}–{clock(float(item["end"]))}"></i>'
            )
        return "".join(parts)

    return (
        '<div class="timeline" aria-label="Episode removal timeline">'
        f'<div class="track reference">{bars(references, "bar")}</div>'
        f'<div class="track predicted">{bars(predicted, "bar")}</div>'
        "</div>"
    )


def render_html(results: dict[str, Any]) -> str:
    episode_map = {
        episode["episode_id"]: episode for episode in results["episodes"]
    }
    runs = results.get("model_runs", [])
    completed_runs = sum(
        run.get("summary", {}).get("execution_passes", 0) for run in runs
    )
    total_cases = sum(
        run.get("summary", {}).get("execution_total", 0) for run in runs
    )
    quality_passes = sum(
        run.get("summary", {}).get("quality_passes", 0) for run in runs
    )
    total_cost = sum(
        float(run.get("summary", {}).get("estimated_cost_usd", 0)) for run in runs
    )

    table_rows = []
    for run in runs:
        model = run["model"]
        summary = run["summary"]
        table_rows.append(
            "<tr>"
            f"<td><strong>{h(model['display_name'])}</strong>"
            f"<small>{h(model['model_id'])}</small></td>"
            f"<td>{h(model['parameters'])}</td>"
            f"<td>{status_badge(summary['overall_status'])}</td>"
            f"<td>{summary['execution_passes']}/{summary['execution_total']}</td>"
            f"<td>{summary['quality_passes']}/{summary['execution_total']}</td>"
            f"<td>{summary['average_f1']:.3f}</td>"
            f"<td>${summary['estimated_cost_usd']:.4f}</td>"
            f"<td>{summary['wall_seconds']:.1f}s</td>"
            "</tr>"
        )

    episode_sections = []
    for episode in results["episodes"]:
        reference_list = "".join(
            "<li>"
            f"<strong>{clock(segment['start'])}–{clock(segment['end'])}</strong> "
            f"{segment['duration']:.2f}s · {h(segment['label'])}"
            "</li>"
            for segment in episode["reference_segments"]
        )
        comparisons = []
        for run in runs:
            result = next(
                (
                    item
                    for item in run["episodes"]
                    if item["episode_id"] == episode["episode_id"]
                ),
                None,
            )
            if not result:
                continue
            model = run["model"]
            metadata = result.get("metadata") or {}
            usage = metadata.get("usage") or {}
            if result.get("metrics"):
                metrics = result["metrics"]
                score = (
                    f"P {metrics['precision']:.3f} · R {metrics['recall']:.3f} · "
                    f"F1 {metrics['f1']:.3f}"
                )
            else:
                score = h(
                    f"{result.get('error', {}).get('type', 'Run failed')}: "
                    f"{result.get('error', {}).get('message', 'No score available')}"
                )
            chunks = (
                f"{metadata.get('completed_chunks', 0)}/"
                f"{metadata.get('chunk_count', '?')} chunks"
            )
            tokens = (
                f"{usage_tokens(usage, 'prompt'):,} input + "
                f"{usage_tokens(usage, 'completion'):,} output tokens"
            )
            comparisons.append(
                '<article class="comparison">'
                "<header>"
                f"<div><h4>{h(model['display_name'])}</h4>"
                f"<small>{h(model['parameters'])} · {h(chunks)} · {h(tokens)}</small></div>"
                f"<div>{status_badge(result['execution_status'], 'RUN')} "
                f"{status_badge(result['quality_status'], 'QUALITY')}</div>"
                "</header>"
                f"<p class=\"score\">{score} · ${result['estimated_cost_usd']:.4f} · "
                f"{result['wall_seconds']:.1f}s</p>"
                f"{timeline(episode['duration'], episode['reference_segments'], result['predicted_segments'])}"
                '<div class="timeline-key"><span class="ref-key">Reference</span>'
                '<span class="pred-key">Detected</span></div>'
                f"{segment_list(result['predicted_segments'])}"
                "</article>"
            )
        episode_sections.append(
            '<section class="episode">'
            f"<h3>Episode {h(episode['episode_id'])}: {h(episode['case'])}</h3>"
            f"<p class=\"muted\">{clock(episode['duration'])} · "
            f"{episode['transcript_segments']:,} transcript segments · "
            f"{episode['transcript_characters']:,} transcript characters</p>"
            "<details><summary>Historical reference removal windows</summary>"
            f'<ol class="reference-list">{reference_list}</ol></details>'
            f'<div class="comparison-grid">{"".join(comparisons)}</div>'
            "</section>"
        )

    generated = results["updated_at"].replace("T", " ").replace("+00:00", " UTC")
    threshold = float(results["quality_f1_threshold"])
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Local LLM Ad Detection Evaluation</title>
<style>
:root{{--bg:#07111f;--panel:#0e1b2d;--panel2:#13243a;--line:#29405d;--text:#eef5ff;
--muted:#9db0c8;--cyan:#45d4d0;--amber:#ffc857;--green:#51d88a;--red:#ff7185;--blue:#5ca9ff}}
*{{box-sizing:border-box}} body{{margin:0;background:radial-gradient(circle at 80% 0,#123052 0,transparent 32%),var(--bg);
color:var(--text);font:15px/1.5 Inter,ui-sans-serif,system-ui,-apple-system,Segoe UI,sans-serif}}
main{{max-width:1440px;margin:auto;padding:40px 24px 72px}} h1{{font-size:clamp(2rem,5vw,4.8rem);
line-height:.95;letter-spacing:-.055em;max-width:900px;margin:18px 0}} h2{{font-size:1.6rem;margin-top:48px}}
h3{{font-size:1.3rem;margin:0}} h4{{margin:0 0 3px}} p{{margin:.5rem 0}}
.eyebrow{{color:var(--cyan);text-transform:uppercase;letter-spacing:.16em;font-weight:800;font-size:.75rem}}
.lead{{max-width:860px;color:#c7d7e9;font-size:1.05rem}} .muted,small{{color:var(--muted)}}
.cards{{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px;margin:30px 0}}
.card,.episode,.comparison,.method{{background:linear-gradient(145deg,rgba(19,36,58,.94),rgba(11,25,42,.94));
border:1px solid var(--line);border-radius:16px;box-shadow:0 18px 50px rgba(0,0,0,.18)}}
.card{{padding:18px}} .card strong{{font-size:1.7rem;display:block}} .card span{{color:var(--muted)}}
.table-wrap{{overflow:auto;border:1px solid var(--line);border-radius:14px}} table{{width:100%;border-collapse:collapse;
min-width:950px;background:rgba(10,23,39,.85)}} th,td{{padding:13px 14px;border-bottom:1px solid var(--line);text-align:left}}
th{{color:#bcd0e8;font-size:.75rem;text-transform:uppercase;letter-spacing:.08em;background:#102139;position:sticky;top:0}}
td small{{display:block;font-size:.72rem;max-width:320px;overflow-wrap:anywhere}} tr:last-child td{{border:0}}
.badge{{display:inline-block;border-radius:99px;padding:3px 8px;font-size:.68rem;font-weight:900;letter-spacing:.08em}}
.badge.pass{{color:#072416;background:var(--green)}} .badge.fail{{color:#3b0911;background:var(--red)}}
.method{{padding:18px 20px;margin-top:18px}} .method code{{color:var(--amber)}} .episode{{padding:22px;margin-top:18px}}
details{{margin:14px 0}} summary{{cursor:pointer;color:#c7d7e9}} .reference-list{{margin:10px 0}}
.comparison-grid{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px;margin-top:18px}}
.comparison{{padding:16px}} .comparison header{{display:flex;gap:12px;justify-content:space-between;align-items:flex-start}}
.score{{font-variant-numeric:tabular-nums;color:#d6e6f7}} .timeline{{display:grid;gap:4px;margin-top:13px}}
.track{{height:9px;background:#07111f;border-radius:99px;position:relative;overflow:hidden}} .bar{{height:100%;position:absolute;border-radius:99px}}
.reference .bar{{background:var(--amber)}} .predicted .bar{{background:var(--cyan)}} .timeline-key{{display:flex;gap:15px;
font-size:.68rem;color:var(--muted);margin:5px 0 8px}} .timeline-key span:before{{content:"";display:inline-block;width:8px;height:8px;
border-radius:50%;margin-right:5px}} .ref-key:before{{background:var(--amber)}} .pred-key:before{{background:var(--cyan)}}
.segments{{padding-left:22px;margin:8px 0 0}} .segments li{{padding:5px 0}} .segments span{{color:var(--muted);margin-left:5px}}
.segments small{{display:block}} .segments .matched{{color:var(--green)}} .segments .extra{{color:var(--red)}}
.note{{border-left:3px solid var(--amber);padding-left:14px;color:#c7d7e9}} footer{{margin-top:40px;color:var(--muted)}}
@media(max-width:900px){{.cards,.comparison-grid{{grid-template-columns:1fr 1fr}}}}
@media(max-width:620px){{main{{padding:24px 14px 50px}}.cards,.comparison-grid{{grid-template-columns:1fr}}
.comparison header{{display:block}}.comparison header>div:last-child{{margin-top:8px}}}}
</style>
</head>
<body><main>
<div class="eyebrow">Podcast Ad Remover · model benchmark</div>
<h1>Local-class LLM ad detection</h1>
<p class="lead">A production-derived comparison of constrained models from 4B through 72B against
Gemini 3.1 Flash Lite. The benchmark evaluates exact removal windows, execution reliability,
cost, and time while keeping cloud models as the product default.</p>
<div class="cards">
<div class="card"><strong>{len(runs)}</strong><span>model configurations</span></div>
<div class="card"><strong>{completed_runs}/{total_cases}</strong><span>completed episode runs</span></div>
<div class="card"><strong>{quality_passes}/{total_cases}</strong><span>quality passes</span></div>
<div class="card"><strong>${total_cost:.4f}</strong><span>estimated OpenRouter cost</span></div>
</div>
<p class="note">Quality pass = complete run with interval F1 ≥ {threshold:.2f}. Historical production
removal windows are a regression reference, not hand-adjudicated ground truth. A failed quality gate
does not by itself prove a detected segment is wrong.</p>
<h2>Model comparison</h2>
<div class="table-wrap"><table><thead><tr><th>Configuration</th><th>Size</th><th>Overall</th>
<th>Execution</th><th>Quality</th><th>Average F1</th><th>Cost</th><th>Time</th></tr></thead>
<tbody>{"".join(table_rows)}</tbody></table></div>
<div class="method"><strong>Consistent test conditions.</strong> Open/local-class runs use an 8K declared
context, 30-second overlap, whitelist classification, reasons disabled, and temperature 0. The standard
Gemini control uses one request and provider-default sampling. OpenRouter models are pinned individually;
there is no fallback cascade. Summary generation is disabled whenever chunking is enabled.</div>
<h2>Episode-by-episode detections</h2>
{"".join(episode_sections)}
<footer>Generated {h(generated)} · Catalogue checked {h(results['catalog_checked_at'])} ·
No transcripts, episode titles, API keys, or raw model responses are embedded in this report.</footer>
</main></body></html>"""


def write_outputs(results: dict[str, Any], results_path: Path, html_path: Path) -> None:
    results["updated_at"] = utc_now()
    results_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(
        json.dumps(results, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    html_path.write_text(render_html(results), encoding="utf-8")


def public_model(model: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in model.items()
        if key
        not in {
            "prompt_price_per_token",
            "completion_price_per_token",
        }
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument(
        "--matrix",
        type=Path,
        default=REPO_ROOT / "scripts" / "local_llm_model_matrix.json",
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=REPO_ROOT / "Documentation" / "LOCAL_LLM_EVALUATION_RESULTS.json",
    )
    parser.add_argument(
        "--html",
        type=Path,
        default=REPO_ROOT / "Documentation" / "LOCAL_LLM_EVALUATION_REPORT.html",
    )
    parser.add_argument("--env-file", type=Path, default=REPO_ROOT / ".env")
    parser.add_argument("--api-key-env", default="OPENROUTER_API_KEY")
    parser.add_argument(
        "--only",
        help="Comma-separated run IDs. Omit to run the complete matrix.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Preserve completed runs not selected by this invocation.",
    )
    parser.add_argument(
        "--delay-seconds",
        type=float,
        default=1.0,
        help="Pause between model configurations to reduce provider bursts.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    api_key = load_env_value(args.env_file, args.api_key_env)
    if not api_key:
        raise SystemExit(
            f"{args.api_key_env} was not found in the environment or {args.env_file}"
        )
    matrix = json.loads(args.matrix.read_text(encoding="utf-8"))
    corpus = json.loads(args.corpus.read_text(encoding="utf-8"))
    selected_ids = (
        {item.strip() for item in args.only.split(",") if item.strip()}
        if args.only
        else None
    )
    selected_models = [
        model
        for model in matrix["models"]
        if selected_ids is None or model["run_id"] in selected_ids
    ]
    if selected_ids:
        missing = selected_ids - {model["run_id"] for model in selected_models}
        if missing:
            raise SystemExit(f"Unknown run IDs: {', '.join(sorted(missing))}")

    existing = None
    if args.resume and args.results.exists():
        existing = json.loads(args.results.read_text(encoding="utf-8"))
    results = build_results(matrix, corpus, existing)
    existing_by_id = {
        run["model"]["run_id"]: run for run in results.get("model_runs", [])
    }
    logging.getLogger().setLevel(logging.WARNING)

    for model_index, model in enumerate(selected_models, start=1):
        print(
            f"[{model_index}/{len(selected_models)}] {model['display_name']}",
            flush=True,
        )
        detector = EvaluationAdDetector(build_settings(model, api_key))
        run = {
            "model": public_model(model),
            "started_at": utc_now(),
            "episodes": [],
        }
        for episode_index, episode in enumerate(corpus["episodes"], start=1):
            print(
                f"  episode {episode['id']} ({episode_index}/{len(corpus['episodes'])})",
                flush=True,
            )
            result = evaluate_episode(
                detector,
                model,
                episode,
                float(matrix["quality_f1_threshold"]),
            )
            run["episodes"].append(result)
            metric_text = (
                f"F1={result['metrics']['f1']:.3f}"
                if result["metrics"]
                else result.get("error", {}).get("type", "failed")
            )
            print(
                f"    {result['execution_status']} / "
                f"{result['quality_status']} / {metric_text}",
                flush=True,
            )
        run["completed_at"] = utc_now()
        run["summary"] = summarize_run(run)
        existing_by_id[model["run_id"]] = run
        matrix_order = {
            item["run_id"]: index for index, item in enumerate(matrix["models"])
        }
        results["model_runs"] = sorted(
            existing_by_id.values(),
            key=lambda item: matrix_order.get(
                item["model"]["run_id"], len(matrix_order)
            ),
        )
        write_outputs(results, args.results, args.html)
        if model_index < len(selected_models) and args.delay_seconds > 0:
            time.sleep(args.delay_seconds)

    print(f"JSON: {args.results}")
    print(f"HTML: {args.html}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
