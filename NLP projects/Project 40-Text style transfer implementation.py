# Install if not already: pip install transformers torch
 
from transformers import pipeline
 
# Load a style transfer model
# Model options: some are fine-tuned T5s, GPT-2, or paraphrasers
# For demo, we'll use a general text2text generation model (T5 paraphrasing variant)
style_transfer = pipeline("text2text-generation", model="prithivida/parrot_paraphraser_on_T5")
 
# Sample sentences and style instructions
input_texts = [
    "The presentation was highly professional and informative.",
    "I am extremely disappointed with the product I received."
]
 
print("🧠 Text Style Transfer Examples:\n")
 
for text in input_texts:
    print(f"🔤 Original: {text}")
    output = style_transfer(f"paraphrase: {text}", max_length=50, num_return_sequences=2)
    for i, o in enumerate(output):
        print(f"  ✍️ Style Variant {i+1}: {o['generated_text']}")
    print()