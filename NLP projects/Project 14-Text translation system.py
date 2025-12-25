# Install if not already: pip install transformers sentencepiece
 
from transformers import MarianMTModel, MarianTokenizer
 
# Choose a language pair model: English to French (you can change this)
model_name = "Helsinki-NLP/opus-mt-en-fr"
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)
 
# Input sentence in English
text = "Artificial Intelligence is transforming the world."
 
# Tokenize and translate
tokens = tokenizer.prepare_seq2seq_batch([text], return_tensors="pt", padding=True)
translated = model.generate(**tokens)
output = tokenizer.decode(translated[0], skip_special_tokens=True)
 
print("🗣️ Original (English):", text)
print("🌍 Translated (French):", output)