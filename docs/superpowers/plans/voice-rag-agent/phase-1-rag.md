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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest backend/tests/test_generate_catalog.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'backend.data.generate_catalog'`

- [ ] **Step 3: Write minimal implementation**

```python
# backend/data/generate_catalog.py
import random
import uuid

CATEGORIES = ["Electronics", "Home & Kitchen", "Apparel", "Books", "Sports", "Toys", "Grocery"]
ADJECTIVES = ["Compact", "Premium", "Portable", "Wireless", "Eco-Friendly", "Heavy-Duty", "Classic"]
NOUNS = ["Blender", "Backpack", "Headphones", "Lamp", "Notebook", "Sneakers", "Kettle", "Charger"]


def generate_catalog(n: int, seed: int = 42) -> list[dict]:
    rng = random.Random(seed)
    items = []
    for _ in range(n):
        adjective = rng.choice(ADJECTIVES)
        noun = rng.choice(NOUNS)
        category = rng.choice(CATEGORIES)
        name = f"{adjective} {noun}"
        items.append({
            "id": str(uuid.uuid4()),
            "name": name,
            "description": f"{name} in the {category} category. Durable build, "
                           f"well-reviewed, ships within 2 days.",
            "price": round(rng.uniform(5.0, 500.0), 2),
            "stock": rng.randint(0, 500),
            "category": category,
        })
    return items


if __name__ == "__main__":
    import json
    catalog = generate_catalog(n=1200, seed=42)
    with open("backend/data/catalog.json", "w") as f:
        json.dump(catalog, f, indent=2)
```

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

## Phase 1 done when

`pytest backend/tests/test_generate_catalog.py backend/tests/test_rag_store.py -v` passes and the manual query in Task 1.3 returns sensible results, including for a Hindi query.
