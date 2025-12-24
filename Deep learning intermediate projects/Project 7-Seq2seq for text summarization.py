# Install if not already: pip install transformers torch
 
from transformers import T5Tokenizer, T5ForConditionalGeneration
 
# Load pre-trained T5 model and tokenizer
model_name = "t5-small"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)
 
# Input long text to summarize
text = """
Artificial Intelligence (AI) refers to the simulation of human intelligence in machines that are programmed to think and learn. 
It is a rapidly evolving field with applications in healthcare, finance, automotive, and more. 
AI systems use data and algorithms to perform tasks such as image recognition, language translation, and decision-making.
"""
 
# Prepend task prefix and tokenize
input_text = "summarize: " + text
inputs = tokenizer.encode(input_text, return_tensors="pt", max_length=512, truncation=True)
 
# Generate summary
summary_ids = model.generate(inputs, max_length=50, min_length=20, length_penalty=2.0, num_beams=4, early_stopping=True)
 
# Decode and print summary
summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
print("📝 Original Text:\n", text)
print("\n✂️ Generated Summary:\n", summary)