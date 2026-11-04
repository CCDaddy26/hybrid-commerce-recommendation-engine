from pathlib import Path

import pandas as pd

from app.data_loader import load_interactions
from app.preprocessor import preprocess_interactions
from app.matrix_builder import build_user_item_matrix
from app.popularity import PopularityRecommender
from app.als_recommender import ImplicitALSRecommender
from app.content_recommender import ContentRecommender
from app.item_metadata import load_item_metadata
from app.hybrid_recommender import HybridRecommender


BASE = Path(__file__).resolve().parent
DATA = BASE / "data"

# --------------------------------------------------
# 1. LOAD DATA
# --------------------------------------------------

interactions = load_interactions(DATA / "sample_interactions.csv")
metadata = load_item_metadata(DATA / "item_metadata.csv")

catalog = metadata.set_index("item_id")

def product_name(item_id):
    if item_id in catalog.index:
        return catalog.loc[item_id, "title"]
    return str(item_id)

def display_results(title, results):
    print(f"\n{title}")
    print("-" * 65)

    if not results:
        print("No recommendations available.")
        return

    for rank, (item_id, score) in enumerate(results, start=1):
        print(
            f"{rank}. {product_name(item_id):35} "
            f"| {item_id} | Score: {score:.4f}"
        )


print("\n" + "=" * 65)
print("       HYBRID COMMERCE RECOMMENDATION ENGINE")
print("=" * 65)

print(f"\nInteractions loaded: {len(interactions)}")
print(f"Products available: {len(metadata)}")

# --------------------------------------------------
# 2. TRAIN THE RECOMMENDATION MODELS
# --------------------------------------------------

print("\nTraining recommendation models...")

preprocessed = preprocess_interactions(interactions)
user_item_matrix = build_user_item_matrix(preprocessed)

popularity = PopularityRecommender()
popularity.fit(interactions)

als = ImplicitALSRecommender(
    factors=32,
    regularization=0.05,
    iterations=15,
    random_state=42
)
als.fit(user_item_matrix, preprocessed, show_progress=False)

content = ContentRecommender()
content.fit(metadata)

hybrid = HybridRecommender(
    als_recommender=als,
    content_recommender=content,
    popularity_recommender=popularity
)

print("All models trained successfully.")

# --------------------------------------------------
# 3. SELECT USER
# --------------------------------------------------

users = sorted(interactions["user_id"].unique())

print("\nAVAILABLE USERS")
print("-" * 65)

for user in users:
    print(f"{user}")

user_id = input("\nEnter user ID (e.g. U101): ").strip().upper()

if user_id not in users:
    print("Unknown user. Using U101.")
    user_id = users[0]

history = interactions.loc[
    interactions["user_id"] == user_id,
    "item_id"
].unique().tolist()

print(f"\nSelected customer: {user_id}")
print("Previously interacted products:")

for item in history:
    print(f"  - {product_name(item)} ({item})")

# --------------------------------------------------
# 4. COMPARE ALL FOUR MODELS
# --------------------------------------------------

display_results(
    "POPULARITY RECOMMENDATIONS",
    popularity.recommend(n=5)
)

display_results(
    "COLLABORATIVE FILTERING — ALS",
    als.recommend(user_id, n=5, filter_already_liked_items=True)
)

display_results(
    "CONTENT-BASED RECOMMENDATIONS",
    content.recommend_for_user(history, n=5)
)

hybrid_results = hybrid.recommend(
    user_id=user_id,
    user_history=history,
    n=5,
    return_breakdown=True
)

print("\nHYBRID RECOMMENDATIONS")
print("-" * 65)

if not hybrid_results:
    print("No recommendations available.")
else:
    for rank, result in enumerate(hybrid_results, start=1):
        print(
            f"\n{rank}. {product_name(result.item_id)} "
            f"({result.item_id})"
        )
        print(f"   Final score: {result.score:.4f}")

        for model, contribution in result.breakdown.items():
            print(f"   {model}: {contribution:.4f}")

# --------------------------------------------------
# 5. NEW USER — PREFERENCE-BASED COLD START
# --------------------------------------------------

print("\n" + "=" * 65)
print("NEW USER COLD-START DEMONSTRATION")
print("=" * 65)

preference = input(
    "\nEnter a new customer's interests "
    "(e.g. wireless ergonomic computer accessories): "
).strip()

if preference:
    new_user_results = hybrid.recommend(
        user_id=None,
        user_history=[],
        preference_text=preference,
        n=5
    )
else:
    new_user_results = hybrid.recommend(
        user_id=None,
        user_history=[],
        n=5
    )

display_results("NEW USER RECOMMENDATIONS", new_user_results)

# --------------------------------------------------
# 6. COLD-START ITEM DEMONSTRATION
# --------------------------------------------------

print("\n" + "=" * 65)
print("COLD-START ITEM DEMONSTRATION")
print("=" * 65)

print("\nNew product: PROD_006 — Ergonomic Vertical Mouse")

similar_items = content.recommend_similar_items(
    "PROD_006",
    n=5,
    exclude_items=["PROD_006"]
)

display_results("PRODUCT SIMILARITY RECOMMENDATIONS", similar_items)

print("\n" + "=" * 65)
print("DEMO COMPLETED")
print("=" * 65)

