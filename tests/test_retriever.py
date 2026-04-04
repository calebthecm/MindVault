"""Tests for hybrid scoring — pure math, no Qdrant or LLM."""

from src.memory.retriever import hybrid_score, _recency_score


def test_hybrid_score_all_ones():
    score = hybrid_score(embedding_sim=1.0, entity_score=1.0, recency=1.0, importance=1.0)
    assert abs(score - 1.0) < 0.001


def test_hybrid_score_embedding_weight():
    score = hybrid_score(embedding_sim=1.0, entity_score=0.0, recency=0.0, importance=0.0)
    assert abs(score - 0.5) < 0.001


def test_hybrid_score_all_zeros():
    score = hybrid_score(embedding_sim=0.0, entity_score=0.0, recency=0.0, importance=0.0)
    assert score == 0.0


def test_recency_score_today():
    from datetime import datetime, timezone
    now_iso = datetime.now(timezone.utc).isoformat()
    score = _recency_score(now_iso)
    assert score > 0.99


def test_recency_score_old():
    score = _recency_score("2020-01-01T00:00:00+00:00")
    assert score < 0.01


def test_recency_score_invalid():
    score = _recency_score("not-a-date")
    assert score == 0.5


def test_hybrid_score_in_range():
    score = hybrid_score(0.8, 0.6, 0.7, 0.9)
    assert 0.0 <= score <= 1.0
