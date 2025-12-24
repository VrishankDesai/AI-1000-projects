# Install if not already: pip install transformers sentencepiece torch
 
from transformers import MarianTokenizer, MarianMTModel
 
# Define source and target language
src_lang = "en"   # English
tgt_lang = "fr"   # French
model_name = f"Helsinki-NLP/opus-mt-{src_lang}-{tgt_lang}"
 
# Load tokenizer and model
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)
 
# Input sentence in source language
text = "Artificial Intelligence is transforming the world."
 
# Tokenize and translate
inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
translated = model.generate(**inputs)
 
# Decode output
output = tokenizer.decode(translated[0], skip_special_tokens=True)
print(f"🌐 Original ({src_lang.upper()}): {text}")
print(f"🔁 Translated ({tgt_lang.upper()}): {output}")