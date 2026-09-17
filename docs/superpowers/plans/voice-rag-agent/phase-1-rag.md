# Phase 1: Inventory Data + RAG Retrieval — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a Python module that, given a text query in English, Hindi or Marathi, returns the top-k most relevant items from a synthetic 1000+ item inventory catalog.

**Architecture:** A seeded generator writes a synthetic catalog to JSON. An `InventoryStore` class embeds one chunk per item with `intfloat/multilingual-e5-small` and upserts into a local persistent Chroma collection; queries embed the user text with the e5 `query:` prefix and return ranked items.

**Tech Stack:** Python 3.10+, ChromaDB, sentence-transformers.

**Spec:** `docs/superpowers/specs/2026-09-17-voice-rag-agent-design.md`

**Index:** [README.md](README.md)

## Isolation

Depends on **nothing**. Runs on CPU. No server, no LLM, no audio. Can be built and fully tested before, after, or in parallel with every other phase.

## Contract produced (other phases rely on this — do not change without updating them)

- Catalog item dict: `{"id": str, "name": str, "description": str, "price": float, "stock": int, "category": str}`
- Retrieved item dict: catalog item plus `{"score": float}`
- `InventoryStore(persist_dir: str)`
- `InventoryStore.load(items: list[dict]) -> None`
- `InventoryStore.query(text: str, top_k: int = 5) -> list[dict]`

Consumed by: Phase 2 (item shape in prompts), Phase 5 (calls `query`).

## Global Constraints

- Chunking: one Chroma chunk per inventory item, no text splitting. *(spec: RAG & inference details)*
- Embedding model: `intfloat/multilingual-e5-small` — must be multilingual so Hindi/Marathi queries match English-described inventory. *(spec: RAG & inference details)*
- Retrieval capped to top-k=3-5 per turn in production use. *(spec: Cost / token optimization)*
- Chroma persists to local Colab disk and is re-embedded fresh each boot — no cross-session durability required. *(spec: Inventory data)*
- Embeddings cached at catalog load; never recomputed except for the live user query. *(spec: Cost / token optimization)*

---

### Task 1.1: Synthetic catalog generator

**Files:**
- Create: `backend/data/generate_catalog.py`
- Create: `backend/data/catalog.json` (generated output, committed as a fixture)
- Test: `backend/tests/test_generate_catalog.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `generate_catalog(n: int, seed: int = 42) -> list[dict]`, each dict shaped `{"id": str, "name": str, "description": str, "price": float, "stock": int, "category": str}`.

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_generate_catalog.py
from backend.data.generate_catalog import generate_catalog

def test_generate_catalog_produces_requested_count():
    items = generate_catalog(n=50, seed=1)
    assert len(items) == 50

def test_generate_catalog_items_have_required_fields():
    items = generate_catalog(n=5, seed=1)
    for item in items:
        assert set(item.keys()) == {"id", "name", "description", "price", "stock", "category"}
        assert isinstance(item["price"], float)
        assert isinstance(item["stock"], int)

def test_generate_catalog_ids_unique():
    items = generate_catalog(n=200, seed=2)
    ids = [item["id"] for item in items]
    assert len(ids) == len(set(ids))

def test_generate_catalog_names_are_distinct():
    # Retrieval is meaningless if the catalog is 1200 copies of 56 names:
    # the top-5 would be near-identical rows the embeddings cannot rank.
    items = generate_catalog(n=1200, seed=42)
    names = [item["name"] for item in items]
    assert len(set(names)) == len(names)

def test_generate_catalog_descriptions_carry_searchable_attributes():
    items = generate_catalog(n=100, seed=3)
    for item in items:
        assert item["category"].split(" ")[0].lower() in item["description"].lower()
        assert len(item["description"].split()) >= 12
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest backend/tests/test_generate_catalog.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'backend.data.generate_catalog'`

- [ ] **Step 3: Write minimal implementation**

```python
# backend/data/generate_catalog.py
"""Synthetic inventory catalog.

Diversity is the whole point here. A naive generator (a handful of adjectives
crossed with a handful of nouns) produces hundreds of items sharing the same
name and description, which makes retrieval look broken: the top-5 comes back
as five indistinguishable rows, and no embedding model can rank identical
text. Every item below gets a unique name and a description carrying its own
searchable attributes.
"""
import random
import uuid

PRODUCTS = {
    "Electronics": [
        ("Wireless Headphones", ["40mm drivers", "active noise cancelling", "open-back"]),
        ("Bluetooth Speaker", ["waterproof", "360-degree sound", "clip-on"]),
        ("USB-C Charger", ["65W GaN", "dual-port", "travel-size"]),
        ("Mechanical Keyboard", ["hot-swappable", "low-profile", "tenkeyless"]),
        ("Webcam", ["1080p", "4K HDR", "auto-framing"]),
        ("Power Bank", ["10000mAh", "20000mAh", "magnetic wireless"]),
    ],
    "Home & Kitchen": [
        ("Blender", ["personal size", "1200W professional", "cordless"]),
        ("Electric Kettle", ["gooseneck", "temperature-control", "glass body"]),
        ("Cast Iron Skillet", ["10-inch", "pre-seasoned", "enamelled"]),
        ("Air Fryer", ["4-quart", "dual-basket", "convection oven"]),
        ("Coffee Grinder", ["burr", "blade", "hand-crank"]),
        ("Desk Lamp", ["LED dimmable", "clamp-mount", "warm-white"]),
    ],
    "Apparel": [
        ("Running Shoes", ["trail", "road", "carbon-plate"]),
        ("Rain Jacket", ["packable", "3-layer shell", "insulated"]),
        ("Cotton T-Shirt", ["heavyweight", "pocket", "long-sleeve"]),
        ("Denim Jacket", ["oversized", "sherpa-lined", "stonewashed"]),
        ("Wool Socks", ["merino hiking", "cushioned crew", "ankle"]),
        ("Baseball Cap", ["unstructured", "mesh-back", "waxed cotton"]),
    ],
    "Books": [
        ("Cookbook", ["weeknight vegetarian", "regional Indian", "baking"]),
        ("Novel", ["literary fiction", "detective", "science fiction"]),
        ("Field Guide", ["birds", "wildflowers", "night sky"]),
        ("Notebook", ["dot-grid", "ruled hardcover", "pocket softcover"]),
        ("Biography", ["scientist", "musician", "explorer"]),
        ("Atlas", ["world", "historical", "road"]),
    ],
    "Sports": [
        ("Yoga Mat", ["cork", "extra-thick", "travel-fold"]),
        ("Dumbbell Set", ["adjustable", "hex rubber", "neoprene"]),
        ("Cricket Bat", ["English willow", "Kashmir willow", "junior"]),
        ("Water Bottle", ["insulated steel", "collapsible", "wide-mouth"]),
        ("Resistance Bands", ["loop set", "tube with handles", "fabric hip"]),
        ("Badminton Racket", ["carbon graphite", "aluminium", "beginner"]),
    ],
    "Toys": [
        ("Building Blocks", ["classic brick", "magnetic tile", "wooden"]),
        ("Puzzle", ["1000-piece landscape", "wooden 3D", "floor"]),
        ("Board Game", ["strategy", "party", "two-player"]),
        ("Remote Control Car", ["off-road", "drift", "mini"]),
        ("Stuffed Animal", ["elephant", "bear", "dinosaur"]),
        ("Art Set", ["watercolour", "coloured pencil", "modelling clay"]),
    ],
    "Grocery": [
        ("Ground Coffee", ["single-origin Ethiopian", "dark roast blend", "decaf"]),
        ("Green Tea", ["sencha", "jasmine", "matcha"]),
        ("Olive Oil", ["extra-virgin", "cold-pressed", "infused"]),
        ("Basmati Rice", ["aged", "brown", "organic"]),
        ("Dark Chocolate", ["70% cocoa", "sea-salt", "orange"]),
        ("Mixed Nuts", ["roasted salted", "raw unsalted", "honey-glazed"]),
    ],
}

BRANDS = ["Aurora", "Nimbus", "Vertex", "Kestrel", "Lumen", "Harbour", "Terra",
          "Orbit", "Sable", "Cobalt", "Juniper", "Marlow", "Anvil", "Sonnet", "Pike"]


def generate_catalog(n: int, seed: int = 42) -> list[dict]:
    rng = random.Random(seed)

    # Build every unique (brand, product, variant, category) combination first,
    # then sample from it, so no two items can ever share a name.
    combos = []
    for category, products in PRODUCTS.items():
        for base, variants in products:
            for variant in variants:
                for brand in BRANDS:
                    combos.append((brand, base, variant, category))
    rng.shuffle(combos)

    if n > len(combos):
        raise ValueError(f"Requested {n} items but only {len(combos)} unique combinations exist")

    items = []
    for brand, base, variant, category in combos[:n]:
        name = f"{brand} {variant.title()} {base}"
        price = round(rng.uniform(5.0, 500.0), 2)
        stock = rng.randint(0, 500)
        availability = "in stock" if stock > 0 else "currently out of stock"
        items.append({
            "id": str(uuid.uuid4()),
            "name": name,
            "description": (
                f"{name} is a {variant} {base.lower()} in the {category} category. "
                f"Priced at ${price}, {availability} with {stock} units on hand. "
                f"Popular with customers looking for a {variant} option."
            ),
            "price": price,
            "stock": stock,
            "category": category,
        })
    return items


if __name__ == "__main__":
    import json
    catalog = generate_catalog(n=1200, seed=42)
    with open("backend/data/catalog.json", "w") as f:
        json.dump(catalog, f, indent=2)
    print(f"Wrote {len(catalog)} items with {len({i['name'] for i in catalog})} unique names")
```

Note the deliberate structure: `15 brands × 7 categories × 6 products × 3 variants = 1,890` unique combinations, sampled down to 1,200. Every name is distinct and every description carries the attributes (`variant`, `category`, price, availability) that retrieval needs as signal.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest backend/tests/test_generate_catalog.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Generate the fixture catalog and commit**

```bash
python -m backend.data.generate_catalog
git add backend/data/generate_catalog.py backend/data/catalog.json backend/tests/test_generate_catalog.py
git commit -m "feat: add synthetic inventory catalog generator"
```

---

### Task 1.2: RAG retrieval module (Chroma + multilingual-e5-small)

**Files:**
- Create: `backend/rag/store.py`
- Create: `backend/rag/__init__.py`
- Test: `backend/tests/test_rag_store.py`
- Modify: `backend/requirements.txt` (create if absent)

**Interfaces:**
- Consumes: catalog item dicts from Task 1.1.
- Produces:
  - `class InventoryStore` with `__init__(self, persist_dir: str)`
  - `InventoryStore.load(self, items: list[dict]) -> None` — embeds and upserts all items
  - `InventoryStore.query(self, text: str, top_k: int = 5) -> list[dict]` — returns items (input shape plus `"score": float`) ranked by relevance

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_rag_store.py
import shutil
import tempfile
import pytest
from backend.rag.store import InventoryStore

SAMPLE_ITEMS = [
    {"id": "1", "name": "Wireless Headphones", "description": "Wireless Headphones in the Electronics category. Noise cancelling.", "price": 89.99, "stock": 20, "category": "Electronics"},
    {"id": "2", "name": "Classic Sneakers", "description": "Classic Sneakers in the Apparel category. Comfortable everyday shoe.", "price": 45.0, "stock": 5, "category": "Apparel"},
    {"id": "3", "name": "Portable Blender", "description": "Portable Blender in the Home & Kitchen category. Great for smoothies.", "price": 30.5, "stock": 0, "category": "Home & Kitchen"},
]


@pytest.fixture
def store():
    tmp_dir = tempfile.mkdtemp()
    s = InventoryStore(persist_dir=tmp_dir)
    s.load(SAMPLE_ITEMS)
    yield s
    shutil.rmtree(tmp_dir, ignore_errors=True)


def test_query_returns_top_k(store):
    results = store.query("noise cancelling headphones", top_k=2)
    assert len(results) == 2


def test_query_returns_most_relevant_first(store):
    results = store.query("I want headphones", top_k=1)
    assert results[0]["id"] == "1"


def test_query_result_has_score(store):
    results = store.query("shoes", top_k=1)
    assert "score" in results[0]


def test_query_works_in_hindi(store):
    results = store.query("क्या आपके पास हेडफ़ोन हैं", top_k=1)
    assert results[0]["id"] == "1"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest backend/tests/test_rag_store.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'backend.rag.store'`

- [ ] **Step 3: Add dependencies**

```text
# backend/requirements.txt
chromadb==0.5.5
sentence-transformers==3.0.1
```

- [ ] **Step 4: Write minimal implementation**

```python
# backend/rag/store.py
import chromadb
from sentence_transformers import SentenceTransformer

EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-small"


class InventoryStore:
    def __init__(self, persist_dir: str):
        self._client = chromadb.PersistentClient(path=persist_dir)
        self._collection = self._client.get_or_create_collection("inventory")
        self._model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    def _embed(self, texts: list[str], is_query: bool) -> list[list[float]]:
        # e5 models require these exact prefixes to perform well.
        prefix = "query: " if is_query else "passage: "
        prefixed = [prefix + t for t in texts]
        return self._model.encode(prefixed, normalize_embeddings=True).tolist()

    def load(self, items: list[dict]) -> None:
        docs = [f"{item['name']}. {item['description']}" for item in items]
        embeddings = self._embed(docs, is_query=False)
        self._collection.upsert(
            ids=[item["id"] for item in items],
            documents=docs,
            embeddings=embeddings,
            metadatas=[
                {
                    "name": item["name"],
                    "price": item["price"],
                    "stock": item["stock"],
                    "category": item["category"],
                }
                for item in items
            ],
        )

    def query(self, text: str, top_k: int = 5) -> list[dict]:
        embedding = self._embed([text], is_query=True)[0]
        result = self._collection.query(query_embeddings=[embedding], n_results=top_k)
        items = []
        for i, item_id in enumerate(result["ids"][0]):
            meta = result["metadatas"][0][i]
            items.append({
                "id": item_id,
                "name": meta["name"],
                "description": result["documents"][0][i],
                "price": meta["price"],
                "stock": meta["stock"],
                "category": meta["category"],
                "score": result["distances"][0][i],
            })
        return items
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pip install -r backend/requirements.txt && pytest backend/tests/test_rag_store.py -v`
Expected: PASS (4 tests)

- [ ] **Step 6: Commit**

```bash
git add backend/rag backend/tests/test_rag_store.py backend/requirements.txt
git commit -m "feat: add Chroma-backed RAG retrieval module with multilingual embeddings"
```

---

### Task 1.3: Load full catalog into persistent Chroma store

**Files:**
- Create: `backend/rag/load_catalog.py`
- Modify: `.gitignore` (create if absent)

**Interfaces:**
- Consumes: `InventoryStore` from Task 1.2, `backend/data/catalog.json` from Task 1.1.
- Produces: a populated Chroma store at `backend/rag/chroma_db/` (gitignored, regenerated on each run).

- [ ] **Step 1: Write the script**

```python
# backend/rag/load_catalog.py
import json
from backend.rag.store import InventoryStore

CATALOG_PATH = "backend/data/catalog.json"
PERSIST_DIR = "backend/rag/chroma_db"

if __name__ == "__main__":
    with open(CATALOG_PATH) as f:
        items = json.load(f)
    store = InventoryStore(persist_dir=PERSIST_DIR)
    store.load(items)
    print(f"Loaded {len(items)} items into {PERSIST_DIR}")
```

- [ ] **Step 2: Run it and verify manually**

Run: `python -m backend.rag.load_catalog`
Expected: prints `Loaded 1200 items into backend/rag/chroma_db`

Run: `python -c "from backend.rag.store import InventoryStore; s = InventoryStore('backend/rag/chroma_db'); print(s.query('cheap sneakers', top_k=3))"`
Expected: prints 3 items, at least one sneaker-related, each carrying a `score`.

- [ ] **Step 3: Gitignore the generated store and commit the script**

```bash
echo "backend/rag/chroma_db/" >> .gitignore
git add backend/rag/load_catalog.py .gitignore
git commit -m "feat: add script to load full catalog into Chroma"
```

---

### Task 1.4: Retrieval quality evaluation (recall@5 and MRR)

Without this, "retrieval works" is an anecdote. With it, it is a number you can defend.

**Files:**
- Create: `backend/rag/eval_queries.json`
- Create: `backend/rag/evaluate.py`
- Test: `backend/tests/test_evaluate.py`

**Interfaces:**
- Consumes: `InventoryStore` from Task 1.2, the loaded Chroma store from Task 1.3.
- Produces: `evaluate(store, queries: list[dict], k: int = 5) -> dict` returning `{"recall_at_k": float, "mrr": float, "n": int, "misses": list[dict]}`

- [ ] **Step 1: Build the labeled query set**

Pick 30 real items out of `backend/data/catalog.json` and write a natural spoken question for each — 10 English, 10 Hindi, 10 Marathi. Multilingual coverage is the point: the cross-lingual embedding claim is central to the architecture, so it needs evidence.

```json
[
  {"query": "do you have any noise cancelling headphones", "expected_id": "<paste a real uuid>", "language": "en"},
  {"query": "I need a cordless blender for smoothies", "expected_id": "<paste a real uuid>", "language": "en"},
  {"query": "क्या आपके पास वाटरप्रूफ स्पीकर है", "expected_id": "<paste a real uuid>", "language": "hi"},
  {"query": "तुमच्याकडे योगा मॅट आहे का", "expected_id": "<paste a real uuid>", "language": "mr"}
]
```

Fill it out to 30 entries. Paste real `id` values from the generated catalog — invented ids will silently score zero.

- [ ] **Step 2: Write the failing test**

```python
# backend/tests/test_evaluate.py
from backend.rag.evaluate import evaluate


class StubStore:
    """Returns a fixed ranking so the metric maths can be checked exactly."""
    def __init__(self, ranking):
        self._ranking = ranking

    def query(self, text, top_k=5):
        return [{"id": item_id} for item_id in self._ranking[:top_k]]


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
```

Add `import pytest` at the top of the file.

- [ ] **Step 3: Run test to verify it fails**

Run: `pytest backend/tests/test_evaluate.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'backend.rag.evaluate'`

- [ ] **Step 4: Write minimal implementation**

```python
# backend/rag/evaluate.py
import json


def evaluate(store, queries: list[dict], k: int = 5) -> dict:
    hits = 0
    reciprocal_ranks = []
    misses = []

    for case in queries:
        results = store.query(case["query"], top_k=k)
        ids = [r["id"] for r in results]
        if case["expected_id"] in ids:
            hits += 1
            rank = ids.index(case["expected_id"]) + 1
            reciprocal_ranks.append(1 / rank)
        else:
            reciprocal_ranks.append(0.0)
            misses.append({"query": case["query"], "language": case.get("language"),
                           "expected_id": case["expected_id"], "got": ids})

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

    print(f"n={result['n']}  recall@5={result['recall_at_k']:.2%}  MRR={result['mrr']:.3f}")

    by_language = {}
    for case in queries:
        by_language.setdefault(case["language"], []).append(case)
    for language, subset in sorted(by_language.items()):
        sub = evaluate(store, subset, k=5)
        print(f"  {language}: recall@5={sub['recall_at_k']:.2%}  (n={sub['n']})")

    if result["misses"]:
        print(f"\n{len(result['misses'])} misses:")
        for miss in result["misses"]:
            print(f"  [{miss['language']}] {miss['query']}")
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest backend/tests/test_evaluate.py -v`
Expected: PASS (4 tests)

- [ ] **Step 6: Run the real evaluation and record the result**

Run: `python -m backend.rag.evaluate`
Expected: a per-language breakdown. **Write the output into `docs/interview-defense.md` §12 (Measurement checklist).** The per-language split is the interesting part — if Hindi and Marathi recall drops sharply against English, that is a finding worth being able to explain, not a failure to hide.

- [ ] **Step 7: Commit**

```bash
git add backend/rag/evaluate.py backend/rag/eval_queries.json backend/tests/test_evaluate.py
git commit -m "feat: add recall@5 and MRR retrieval evaluation across three languages"
```

---

## Phase 1 done when

`pytest backend/tests/test_generate_catalog.py backend/tests/test_rag_store.py backend/tests/test_evaluate.py -v` passes, the manual query in Task 1.3 returns sensible results including for a Hindi query, and `python -m backend.rag.evaluate` prints a recorded recall@5 figure per language.
