# Install if not already: pip install transformers torch
 
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
 
# Load pre-trained GPT-2 tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")
 
# Set model to evaluation mode
model.eval()
 
# Provide a text prompt
prompt = "Once upon a time in a distant galaxy,"
input_ids = tokenizer.encode(prompt, return_tensors="pt")
 
# Generate continuation
with torch.no_grad():
    output = model.generate(
        input_ids,
        max_length=100,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.9,
        early_stopping=True
    )
 
# Decode and print generated text
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print("📝 Generated Text:\n")
print(generated_text)