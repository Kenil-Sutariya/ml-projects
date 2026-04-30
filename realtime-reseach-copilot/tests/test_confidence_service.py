"""
test_confidence_service.py — Tests for ConfidenceService (Milestone 5)
"""

import pytest

from app.models.schemas import SourceResult
from app.services.confidence_service import ConfidenceService


def make_source(source_type: str = "wikipedia", content: str = "x" * 800) -> SourceResult:
    return SourceResult(title="Test", content=content, source_type=source_type)


class TestConfidenceService:
    def setup_method(self):
        self.service = ConfidenceService()

    def test_no_sources_returns_low_score(self):
        assert self.service.calculate([]) == pytest.approx(0.10)

    def test_score_is_float_in_range(self):
        score = self.service.calculate([make_source()])
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_more_sources_gives_higher_score(self):
        one = self.service.calculate([make_source()])
        four = self.service.calculate([make_source()] * 4)
        assert four > one

    def test_diverse_sources_give_higher_score_than_single_type(self):
        same_type = [make_source("wikipedia")] * 3
        diverse = [make_source("wikipedia"), make_source("web"), make_source("private_kb")]
        assert self.service.calculate(diverse) > self.service.calculate(same_type)

    def test_score_never_exceeds_1(self):
        many = [make_source("wikipedia"), make_source("web"), make_source("private_kb")] * 10
        assert self.service.calculate(many) <= 1.0

    def test_four_plus_sources_multiple_types_scores_above_0_7(self):
        sources = (
            [make_source("wikipedia")] * 2
            + [make_source("web")] * 2
            + [make_source("private_kb")] * 2
        )
        assert self.service.calculate(sources) >= 0.7
