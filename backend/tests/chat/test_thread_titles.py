from app.chat.thread_titles import DEFAULT_THREAD_TITLE, derive_thread_title, MAX_THREAD_TITLE_LENGTH


def test_derive_thread_title_collapses_whitespace() -> None:
    assert derive_thread_title("  Compare   NVDA   and   AMD  ") == "Compare NVDA and AMD"


def test_derive_thread_title_returns_default_for_blank_input() -> None:
    assert derive_thread_title("   ") == DEFAULT_THREAD_TITLE


def test_derive_thread_title_truncates_long_prompts() -> None:
    prompt = "word " * 100
    title = derive_thread_title(prompt)

    assert len(title) <= MAX_THREAD_TITLE_LENGTH
    assert title.endswith("…")


def test_derive_thread_title_keeps_short_prompts() -> None:
    prompt = "Summarize Apple risk factors from the latest 10-K"
    assert derive_thread_title(prompt) == prompt
