# Install if not already: pip install spacy
# Download the English model: python -m spacy download en_core_web_sm
 
import spacy
 
# Load English NLP model
nlp = spacy.load("en_core_web_sm")
 
# Sample sentence
text = "The quick brown fox jumps over the lazy dog near the riverbank."
 
# Process text
doc = nlp(text)
 
# Print each word with its POS tag and explanation
print("Word\t\tPOS\t\tExplanation")
print("-" * 40)
for token in doc:
    print(f"{token.text:<10}\t{token.pos_:<10}\t{spacy.explain(token.pos_)}")