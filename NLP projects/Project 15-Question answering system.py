# Install if not already: pip install transformers torch
 
from transformers import pipeline
 
# Load Hugging Face QA pipeline
qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")
 
# Context paragraph (can be replaced with dynamic text/documents)
context = """
Artificial Intelligence (AI) refers to the simulation of human intelligence in machines.
These machines are programmed to think like humans and mimic their actions. 
AI is being used in various fields including healthcare, finance, and transportation.
"""
 
# Sample questions
questions = [
    "What does AI refer to?",
    "Where is AI being used?",
    "What do machines do in AI?"
]
 
print("🧠 QA Results:\n")
for question in questions:
    result = qa_pipeline(question=question, context=context)
    print(f"❓ Question: {question}")
    print(f"✅ Answer: {result['answer']}\n")