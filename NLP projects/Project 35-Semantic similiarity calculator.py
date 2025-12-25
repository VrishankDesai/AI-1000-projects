# Install if not already: pip install sentence-transformers
 
from sentence_transformers import SentenceTransformer, util
 
# Load a pretrained SBERT model
model = SentenceTransformer('all-MiniLM-L6-v2')
 
# Example text pairs
sentence_pairs = [
    ("The cat sat on the mat.", "The feline rested on the rug."),
    ("How do I bake a cake?", "What is the recipe for making a cake?"),
    ("I love playing the guitar.", "Bananas are my favorite fruit.")
]
 
print("🧠 Semantic Similarity Scores:\n")
 
for sent1, sent2 in sentence_pairs:
    embedding1 = model.encode(sent1, convert_to_tensor=True)
    embedding2 = model.encode(sent2, convert_to_tensor=True)
 
    similarity_score = util.cos_sim(embedding1, embedding2).item()
    print(f"🔗 \"{sent1}\"\n    \"{sent2}\"\n    → Similarity: {similarity_score:.4f}\n")