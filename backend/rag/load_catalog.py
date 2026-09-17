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
