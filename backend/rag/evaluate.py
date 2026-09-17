"""Retrieval quality evaluation: recall@k and MRR.

Two kinds of judgment, because the catalog carries ~14 brands of each
product+variant:

- `expected_id` — an exact item. Only meaningful for brand-specific queries
  ("the Juniper cordless blender"), where exactly one item is correct.
- `expected_name_contains` — a product-type judgment, a string or a list of
  acceptable strings. For a natural spoken query ("do you have a cordless
  blender"), any brand of that product is a correct answer, so scoring
  against one arbitrary UUID would measure the catalog's brand duplication
  rather than the retriever's quality.
- `expected_category` — a category-level judgment. "I want a book" is
  correctly answered by a novel, a biography or a cookbook; demanding a
  specific product type there would score a correct retrieval as a miss.

Using exact-ID matching for generic queries would cap recall@5 at roughly
5/14 no matter how well retrieval worked.
"""
import json


def _matches(case: dict, result: dict) -> bool:
    if "expected_id" in case:
        return result["id"] == case["expected_id"]

    if "expected_name_contains" in case:
        expected = case["expected_name_contains"]
        needles = [expected] if isinstance(expected, str) else expected
        name = result.get("name", "").lower()
        return any(needle.lower() in name for needle in needles)

    if "expected_category" in case:
        return result.get("category") == case["expected_category"]

    raise ValueError(
        "Query needs expected_id, expected_name_contains or expected_category: "
        f"{case}"
    )


def _is_hit(case: dict, results: list[dict]) -> int | None:
    """Return the 1-based rank of the first correct result, or None."""
    for rank, result in enumerate(results, start=1):
        if _matches(case, result):
            return rank
    return None


def evaluate(store, queries: list[dict], k: int = 5) -> dict:
    hits = 0
    reciprocal_ranks = []
    misses = []

    for case in queries:
        results = store.query(case["query"], top_k=k)
        rank = _is_hit(case, results)
        if rank is not None:
            hits += 1
            reciprocal_ranks.append(1 / rank)
        else:
            reciprocal_ranks.append(0.0)
            misses.append({
                "query": case["query"],
                "language": case.get("language"),
                "expected": (case.get("expected_id")
                             or case.get("expected_name_contains")
                             or case.get("expected_category")),
                "got": [r.get("name", r["id"]) for r in results],
            })

    n = len(queries)
    return {
        "recall_at_k": hits / n if n else 0.0,
        "mrr": sum(reciprocal_ranks) / n if n else 0.0,
        "n": n,
        "misses": misses,
    }


if __name__ == "__main__":
    from backend.rag.store import InventoryStore

    with open("backend/rag/eval_queries.json") as f:
        queries = json.load(f)

    store = InventoryStore(persist_dir="backend/rag/chroma_db")
    result = evaluate(store, queries, k=5)

    print(f"overall  n={result['n']}  recall@5={result['recall_at_k']:.1%}  "
          f"MRR={result['mrr']:.3f}")

    by_language = {}
    for case in queries:
        by_language.setdefault(case["language"], []).append(case)
    for language, subset in sorted(by_language.items()):
        sub = evaluate(store, subset, k=5)
        print(f"  {language}: recall@5={sub['recall_at_k']:.1%}  "
              f"MRR={sub['mrr']:.3f}  (n={sub['n']})")

    if result["misses"]:
        print(f"\n{len(result['misses'])} misses:")
        for miss in result["misses"]:
            print(f"  [{miss['language']}] {miss['query']}")
            print(f"      expected: {miss['expected']}")
            print(f"      got: {miss['got'][0]}")
