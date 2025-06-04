from sentence_transformers import SentenceTransformer, util

# Embedder model
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Few-shot examples
few_shot_db = {
    "how to optimize": "Use gradient clipping, weight decay, and learning rate scheduling.",
    "train transformer": "Make sure to use a warmup scheduler and layer norm.",
    "debug model": "Check for vanishing gradients or incorrect loss scaling."
}
few_shot_keys = list(few_shot_db.keys())
few_shot_embeddings = embedder.encode(few_shot_keys, convert_to_tensor=True)

def few_shot_pattern_matcher(input_data: str) -> str:
    input_embedding = embedder.encode(input_data, convert_to_tensor=True)
    similarity_scores = util.pytorch_cos_sim(input_embedding, few_shot_embeddings)
    best_match_idx = similarity_scores.argmax().item()
    return few_shot_db[few_shot_keys[best_match_idx]]
