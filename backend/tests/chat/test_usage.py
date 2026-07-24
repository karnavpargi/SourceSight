from app.chat.usage import TurnUsage


def test_turn_usage_aggregates_across_calls() -> None:
    usage = TurnUsage()
    usage.add_model_usage(input_tokens=100, output_tokens=20)
    usage.record_embedding()
    usage.record_passages(5)
    usage.add_model_usage(input_tokens=50, output_tokens=10)
    usage.record_correction()
    fields = usage.as_log_fields()
    assert fields == {
        "model_calls": 2,
        "embedding_calls": 1,
        "input_tokens": 150,
        "output_tokens": 30,
        "passages": 5,
        "corrections": 1,
    }
