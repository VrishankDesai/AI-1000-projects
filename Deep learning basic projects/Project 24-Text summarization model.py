# Install if not already: pip install transformers torch
 
from transformers import pipeline
 
# Load the summarization pipeline
summarizer = pipeline("summarization", model="t5-small", tokenizer="t5-small")
 
# Input text (can be a paragraph or document)
text = """
Artificial Intelligence (AI) is rapidly transforming our world. 
From self-driving cars and personalized recommendations to virtual assistants and automated customer service, 
AI applications are everywhere. The technology uses algorithms and large datasets to learn patterns, 
make decisions, and even generate human-like text and images. While AI offers immense potential, 
it also raises ethical concerns around bias, job displacement, and privacy. 
Understanding how AI works is becoming increasingly important for both individuals and organizations.
"""
 
# Generate summary
summary = summarizer("summarize: " + text, max_length=50, min_length=20, do_sample=False)
 
# Print result
print("🔍 Original Text:\n", text, "\n")
print("📝 Summary:\n", summary[0]['summary_text'])