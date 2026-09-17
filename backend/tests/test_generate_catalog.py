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
