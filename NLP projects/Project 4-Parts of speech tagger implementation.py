# Install if not already: pip install spacy
# Then download model: python -m spacy download en_core_web_sm
 
import spacy
 
# Load the English SpaCy model
nlp = spacy.load("en_core_web_sm")
 
# Example sentence or paragraph
text = "The quick brown fox jumps over the lazy dog near the riverbank."
 
# Process the text
doc = nlp(text)
 
# Print each token with its POS tag and detailed tag
print("🧠 Part-of-Speech Tagging:")
print(f"{'Token':<15} {'POS':<10} {'Tag':<10} {'Explanation'}")
print("-" * 60)
for token in doc:
    print(f"{token.text:<15} {token.pos_:<10} {token.tag_:<10} {spacy.explain(token.tag_)}")