import pytest

from backend.rag.evaluate import evaluate


class StubStore:
    """Returns a fixed ranking so the metric maths can be checked exactly."""

    def __init__(self, ranking, category="Electronics"):
        self._ranking = ranking
        self._category = category

    def query(self, text, top_k=5):
        return [{"id": item_id, "name": item_id, "category": self._category}
                for item_id in self._ranking[:top_k]]


def test_recall_counts_hit_anywhere_in_top_k():
    store = StubStore(["a", "b", "c", "d", "e"])
    result = evaluate(store, [{"query": "q", "expected_id": "e", "language": "en"}], k=5)
    assert result["recall_at_k"] == 1.0


def test_recall_counts_miss_outside_top_k():
    store = StubStore(["a", "b", "c", "d", "e"])
    result = evaluate(store, [{"query": "q", "expected_id": "zzz", "language": "en"}], k=5)
    assert result["recall_at_k"] == 0.0
    assert len(result["misses"]) == 1


def test_mrr_uses_reciprocal_of_rank():
    store = StubStore(["a", "b", "c"])
    result = evaluate(store, [{"query": "q", "expected_id": "c", "language": "en"}], k=5)
    assert result["mrr"] == pytest.approx(1 / 3)


def test_reports_count():
    store = StubStore(["a"])
    queries = [{"query": "q", "expected_id": "a", "language": "en"}] * 4
    assert evaluate(store, queries, k=5)["n"] == 4


def test_name_contains_judgment_matches_any_brand():
    store = StubStore(["Juniper Cordless Blender"])
    result = evaluate(
        store,
        [{"query": "cordless blender", "expected_name_contains": "Cordless Blender",
          "language": "en"}],
        k=5,
    )
    assert result["recall_at_k"] == 1.0


def test_name_contains_judgment_is_case_insensitive():
    store = StubStore(["Juniper Cordless Blender"])
    result = evaluate(
        store,
        [{"query": "q", "expected_name_contains": "cordless blender", "language": "en"}],
        k=5,
    )
    assert result["recall_at_k"] == 1.0


def test_name_contains_accepts_a_list_of_alternatives():
    store = StubStore(["Lumen Musician Biography"])
    result = evaluate(
        store,
        [{"query": "I want a book",
          "expected_name_contains": ["Novel", "Biography", "Cookbook"],
          "language": "en"}],
        k=5,
    )
    assert result["recall_at_k"] == 1.0


def test_category_judgment_matches_on_category():
    store = StubStore(["Lumen Musician Biography"], category="Books")
    result = evaluate(
        store,
        [{"query": "I want a book", "expected_category": "Books", "language": "en"}],
        k=5,
    )
    assert result["recall_at_k"] == 1.0


def test_category_judgment_misses_on_wrong_category():
    store = StubStore(["Pike Dot-Grid Notebook"], category="Books")
    result = evaluate(
        store,
        [{"query": "keyboard", "expected_category": "Electronics", "language": "en"}],
        k=5,
    )
    assert result["recall_at_k"] == 0.0


def test_query_without_a_judgment_raises():
    store = StubStore(["a"])
    with pytest.raises(ValueError):
        evaluate(store, [{"query": "q", "language": "en"}], k=5)
