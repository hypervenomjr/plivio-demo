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

# Per-category price bands. A single 5-500 range across everything produces
# $300 chocolate bars, which reads as broken during a live demo even though
# retrieval is working fine.
PRICE_RANGES = {
    "Electronics": (19.99, 499.0),
    "Home & Kitchen": (14.99, 299.0),
    "Apparel": (12.99, 199.0),
    "Books": (4.99, 59.0),
    "Sports": (9.99, 249.0),
    "Toys": (7.99, 129.0),
    "Grocery": (2.99, 39.0),
}


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
        low, high = PRICE_RANGES[category]
        price = round(rng.uniform(low, high), 2)
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
