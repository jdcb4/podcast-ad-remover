from scripts.evaluate_local_llms import (
    interval_metrics,
    render_html,
    safe_error,
    summarize_run,
)


def test_interval_metrics_use_duration_overlap():
    metrics = interval_metrics(
        [{"start": 0, "end": 20}, {"start": 30, "end": 40}],
        [{"start": 10, "end": 35}],
    )

    assert metrics == {
        "precision": 0.5,
        "recall": 0.6,
        "f1": 0.5455,
        "predicted_seconds": 30.0,
        "reference_seconds": 25.0,
        "overlap_seconds": 15.0,
    }


def test_safe_error_redacts_credentials():
    error = safe_error(RuntimeError("Bearer local-test-token-value failed"))

    assert error["type"] == "RuntimeError"
    assert "supersecretvalue" not in error["message"]
    assert "[redacted]" in error["message"]


def test_summarize_run_requires_every_episode_to_pass_quality():
    run = {
        "episodes": [
            {
                "execution_status": "pass",
                "quality_status": "pass",
                "metrics": {"f1": 0.9},
                "estimated_cost_usd": 0.01,
                "wall_seconds": 2,
            },
            {
                "execution_status": "pass",
                "quality_status": "fail",
                "metrics": {"f1": 0.5},
                "estimated_cost_usd": 0.02,
                "wall_seconds": 3,
            },
        ]
    }

    summary = summarize_run(run)

    assert summary["execution_passes"] == 2
    assert summary["quality_passes"] == 1
    assert summary["average_f1"] == 0.7
    assert summary["overall_status"] == "fail"


def test_html_report_contains_detections_but_not_transcripts():
    results = {
        "updated_at": "2026-07-24T00:00:00+00:00",
        "catalog_checked_at": "2026-07-24",
        "quality_f1_threshold": 0.7,
        "episodes": [
            {
                "episode_id": 1,
                "case": "Short",
                "duration": 100,
                "transcript_segments": 2,
                "transcript_characters": 99,
                "reference_segments": [
                    {
                        "start": 0,
                        "end": 10,
                        "duration": 10,
                        "label": "Promo",
                    }
                ],
            }
        ],
        "model_runs": [
            {
                "model": {
                    "run_id": "test",
                    "display_name": "Test model",
                    "model_id": "test/model",
                    "parameters": "7B",
                },
                "summary": {
                    "overall_status": "pass",
                    "execution_passes": 1,
                    "execution_total": 1,
                    "quality_passes": 1,
                    "average_f1": 1,
                    "estimated_cost_usd": 0.001,
                    "wall_seconds": 1,
                },
                "episodes": [
                    {
                        "episode_id": 1,
                        "execution_status": "pass",
                        "quality_status": "pass",
                        "metrics": {"precision": 1, "recall": 1, "f1": 1},
                        "predicted_segments": [
                            {
                                "start": 0,
                                "end": 10,
                                "duration": 10,
                                "label": "Non-Content",
                                "match": "matched",
                                "best_reference_index": 0,
                                "overlap_seconds": 10,
                            }
                        ],
                        "metadata": {
                            "chunk_count": 1,
                            "completed_chunks": 1,
                            "usage": {
                                "prompt_tokens": 100,
                                "completion_tokens": 10,
                            },
                        },
                        "estimated_cost_usd": 0.001,
                        "wall_seconds": 1,
                    }
                ],
            }
        ],
    }

    report = render_html(results)

    assert "Test model" in report
    assert "0:00–0:10" in report
    assert "transcript text" not in report
