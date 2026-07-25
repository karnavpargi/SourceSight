import pytest

from app.chat.usage import TurnUsage


def test_turn_usage_aggregates_across_calls() -> None:
    usage = TurnUsage()
    usage.add_model_usage(
        stage="synthesis",
        model="m",
        input_tokens=100,
        output_tokens=20,
    )
    usage.record_embedding()
    usage.record_passages(5)
    usage.add_model_usage(
        stage="synthesis",
        model="m",
        input_tokens=50,
        output_tokens=10,
    )
    usage.record_correction()
    fields = usage.as_log_fields()
    assert fields == {
        "model_calls": 2,
        "embedding_calls": 1,
        "input_tokens": 150,
        "output_tokens": 30,
        "passages": 5,
        "retrieval_expanded": False,
        "corrections": 1,
        "stages": {
            "synthesis": {
                "model": "m",
                "calls": 2,
                "input_tokens": 150,
                "output_tokens": 30,
            },
        },
        "route": None,
        "budget_profile": None,
    }


def test_turn_usage_tracks_models_by_stage() -> None:
    usage = TurnUsage()
    usage.add_model_usage(
        stage="router",
        model="gemini-2.0-flash-lite",
        input_tokens=100,
        output_tokens=20,
    )
    usage.add_model_usage(
        stage="synthesis",
        model="gemini-3.5-flash-lite",
        input_tokens=500,
        output_tokens=100,
    )
    fields = usage.as_log_fields()
    assert fields["model_calls"] == 2
    assert fields["stages"]["router"]["model"] == "gemini-2.0-flash-lite"
    assert fields["stages"]["synthesis"]["input_tokens"] == 500


def test_estimated_cost_requires_every_exact_model_price() -> None:
    usage = TurnUsage()
    usage.add_model_usage(
        stage="router",
        model="unknown",
        input_tokens=100,
        output_tokens=20,
    )
    assert usage.estimated_cost_usd({}) is None


def test_estimated_cost_uses_per_million_rates() -> None:
    usage = TurnUsage()
    usage.add_model_usage(
        stage="synthesis",
        model="m",
        input_tokens=1_000_000,
        output_tokens=1_000_000,
    )
    assert usage.estimated_cost_usd({"m": (0.30, 2.50)}) == pytest.approx(2.80)
