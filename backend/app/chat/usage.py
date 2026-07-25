from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class StageUsage:
    model: str
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class TurnUsage:
    stages: dict[str, StageUsage] = field(default_factory=dict)
    embedding_calls: int = 0
    passages: int = 0
    corrections: int = 0
    route: str | None = None
    budget_profile: str | None = None

    def add_model_usage(
        self,
        *,
        stage: str,
        model: str,
        input_tokens: int | None,
        output_tokens: int | None,
    ) -> None:
        current = self.stages.get(stage)
        if current is None:
            current = StageUsage(model=model)
            self.stages[stage] = current
        if current.model != model:
            raise ValueError(f"stage {stage!r} used multiple models")
        current.calls += 1
        current.input_tokens += input_tokens or 0
        current.output_tokens += output_tokens or 0

    def estimated_cost_usd(
        self,
        prices: dict[str, tuple[float, float]],
    ) -> float | None:
        if any(stage.model not in prices for stage in self.stages.values()):
            return None
        return sum(
            (
                stage.input_tokens * prices[stage.model][0]
                + stage.output_tokens * prices[stage.model][1]
            )
            / 1_000_000
            for stage in self.stages.values()
        )

    def record_embedding(self) -> None:
        self.embedding_calls += 1

    def record_passages(self, count: int) -> None:
        self.passages = max(self.passages, count)

    def record_correction(self) -> None:
        self.corrections += 1

    @property
    def model_calls(self) -> int:
        return sum(stage.calls for stage in self.stages.values())

    @property
    def input_tokens(self) -> int:
        return sum(stage.input_tokens for stage in self.stages.values())

    @property
    def output_tokens(self) -> int:
        return sum(stage.output_tokens for stage in self.stages.values())

    def as_log_fields(self) -> dict[str, object]:
        return {
            "model_calls": self.model_calls,
            "embedding_calls": self.embedding_calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "passages": self.passages,
            "corrections": self.corrections,
            "stages": {
                name: {
                    "model": stage.model,
                    "calls": stage.calls,
                    "input_tokens": stage.input_tokens,
                    "output_tokens": stage.output_tokens,
                }
                for name, stage in self.stages.items()
            },
            "route": self.route,
            "budget_profile": self.budget_profile,
        }
