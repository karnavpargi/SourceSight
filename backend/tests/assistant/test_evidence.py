from datetime import date
from uuid import uuid4

from app.assistant.evidence import EvidenceRegistry
from app.retrieval.types import SourcePassage


def _passage(content: str = "AWS income", ticker: str = "AMZN") -> SourcePassage:
    return SourcePassage(
        chunk_id=uuid4(),
        document_id=uuid4(),
        chunk_index=0,
        content=content,
        section="Item 8",
        page=None,
        ticker=ticker,
        company_name="Amazon",
        form_type="10-K",
        fiscal_year=2024,
        accession_number="0001",
        filing_date=date(2025, 1, 31),
        report_date=None,
        source_url="https://example.com",
        score=1.0,
    )


def test_registry_assigns_aliases_and_dedupes() -> None:
    registry = EvidenceRegistry(max_passages=8)
    p1 = _passage("one")
    p2 = _passage("two")
    first = registry.register([p1, p2, p1])
    assert [e.alias for e in first] == ["E1", "E2"]
    assert first[0].content == "one"
    assert first[0].ticker == "AMZN"
    assert first[0].fiscal_year == 2024
    assert first[0].section == "Item 8"
    assert "chunk_id" not in first[0].model_dump()
    assert registry.resolve("E1").chunk_id == p1.chunk_id


def test_registry_truncates_compact_content() -> None:
    long_content = "x" * 2000
    registry = EvidenceRegistry()
    passage = _passage(content=long_content)

    compact_list = registry.register([passage])
    assert len(compact_list) == 1
    compact = compact_list[0]

    # Compact content should be hard-capped to ~1200 characters with an ellipsis.
    assert len(compact.content) <= 1200
    assert compact.content.endswith("…")
    # Full passage content is still available via the registry.
    assert registry.resolve(compact.alias).content == long_content


def test_registry_enforces_max_passages() -> None:
    registry = EvidenceRegistry(max_passages=2)
    passages = [_passage(f"c{i}") for i in range(5)]
    compact = registry.register(passages)
    assert len(compact) == 2
    assert len(registry.all_passages()) == 2


def test_registry_rejects_unknown_alias() -> None:
    registry = EvidenceRegistry()
    try:
        registry.resolve("E99")
        raise AssertionError("expected KeyError")
    except KeyError:
        pass


def test_registry_compact_dump_returns_all_compact_rows() -> None:
    registry = EvidenceRegistry()
    p1 = _passage("one")
    p2 = _passage("two")
    registry.register([p1, p2])

    compact = registry.compact_dump()
    aliases = [row.alias for row in compact]
    assert aliases == ["E1", "E2"]
    # compact_dump should use truncated content, not the full passage for long inputs.
    long_passage = _passage(content="x" * 2000)
    registry = EvidenceRegistry()
    registry.register([long_passage])
    dumped = registry.compact_dump()
    assert len(dumped) == 1
    assert len(dumped[0].content) <= 1200
