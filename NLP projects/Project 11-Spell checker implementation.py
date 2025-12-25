# Install if not already: pip install textblob
# And run once: python -m textblob.download_corpora
 
from textblob import TextBlob
 
# Example sentences with spelling mistakes
sentences = [
    "I realy like this librarry.",
    "Ths projct is relly amzing!",
    "Welcom to the wrld of NLP.",
]
 
print("🔍 Spell Checker Results:\n")
for sentence in sentences:
    blob = TextBlob(sentence)
    corrected = blob.correct()
    print(f"❌ Original:  {sentence}")
    print(f"✅ Corrected: {corrected}\n")