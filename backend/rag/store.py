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
