import shutil
import tempfile

import pytest

from backend.rag.store import InventoryStore

SAMPLE_ITEMS = [
    {"id": "1", "name": "Aurora Active Noise Cancelling Wireless Headphones",
     "description": "Aurora Active Noise Cancelling Wireless Headphones is an active noise cancelling wireless headphones in the Electronics category.",
     "price": 89.99, "stock": 20, "category": "Electronics"},
    {"id": "2", "name": "Lumen Road Running Shoes",
     "description": "Lumen Road Running Shoes is a road running shoes in the Apparel category.",
     "price": 45.0, "stock": 5, "category": "Apparel"},
    {"id": "3", "name": "Cobalt Cordless Blender",
     "description": "Cobalt Cordless Blender is a cordless blender in the Home & Kitchen category. Great for smoothies.",
     "price": 30.5, "stock": 0, "category": "Home & Kitchen"},
]


@pytest.fixture(scope="module")
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


def test_query_result_carries_metadata(store):
    results = store.query("blender for smoothies", top_k=1)
    item = results[0]
    assert item["id"] == "3"
    assert item["name"] == "Cobalt Cordless Blender"
    assert item["price"] == 30.5
    assert item["stock"] == 0
    assert item["category"] == "Home & Kitchen"


def test_query_works_in_hindi(store):
    # The cross-lingual claim: a Hindi query must match an English listing.
    results = store.query("क्या आपके पास हेडफ़ोन हैं", top_k=1)
    assert results[0]["id"] == "1"


def test_query_works_in_marathi(store):
    results = store.query("तुमच्याकडे धावण्याचे बूट आहेत का", top_k=1)
    assert results[0]["id"] == "2"
