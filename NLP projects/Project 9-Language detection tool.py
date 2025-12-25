# Install if not already: pip install langdetect
 
from langdetect import detect, detect_langs
 
# Sample texts in different languages
texts = [
    "This is an English sentence.",
    "Ceci est une phrase française.",
    "Dies ist ein deutscher Satz.",
    "これは日本語の文章です。",
    "Esto es una oración en español.",
    "Это предложение на русском языке."
]
 
print("🧠 Detected Languages:\n")
for i, text in enumerate(texts, start=1):
    language = detect(text)
    confidence = detect_langs(text)[0]
    print(f"Text {i}: {text}")
    print(f"🔍 Detected Language: {language} (Confidence: {confidence.prob:.2f})\n")