from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TurnUsage:
    model_calls: int = 0
    embedding_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    passages: int = 0
    corrections: int = 0

    def add_model_usage(
        self,
        *,
        input_tokens: int | None,
        output_tokens: int | None,
    ) -> None:
        self.model_calls += 1
        if input_tokens is not None:
            self.input_tokens += input_tokens
        if output_tokens is not None:
            self.output_tokens += output_tokens

    def record_embedding(self) -> None:
        self.embedding_calls += 1

    def record_passages(self, count: int) -> None:
        self.passages = max(self.passages, count)

    def record_correction(self) -> None:
        self.corrections += 1

    def as_log_fields(self) -> dict[str, int]:
        return {
            "model_calls": self.model_calls,
            "embedding_calls": self.embedding_calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "passages": self.passages,
            "corrections": self.corrections,
        }
