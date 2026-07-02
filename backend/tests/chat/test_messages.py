import pytest

from app.chat.messages import extract_latest_user_text


def test_extract_latest_user_text_from_parts() -> None:
    text = extract_latest_user_text(
        [
            {"role": "assistant", "content": "Earlier answer"},
            {
                "role": "user",
                "parts": [{"type": "text", "text": "AWS operating income"}],
            },
        ]
    )
    assert text == "AWS operating income"


def test_extract_latest_user_text_from_content_field() -> None:
    text = extract_latest_user_text([{"role": "user", "content": "  iPhone revenue  "}])
    assert text == "iPhone revenue"


def test_extract_latest_user_text_raises_when_missing_user_message() -> None:
    with pytest.raises(ValueError, match="No user message"):
        extract_latest_user_text([{"role": "assistant", "content": "No question here"}])
