import spacy
import pandas as pd
import re
from collections import Counter
 
# 1. Load a pre-trained NLP model for named entity recognition (NER)
nlp = spacy.load("en_core_web_sm")
 
# 2. Sample financial document (could be extracted from a PDF or text file)
financial_text = """
Apple Inc. reported a quarterly revenue of $123.9 billion, a 10% increase year-over-year.
The company also declared a dividend of $0.22 per share, which will be paid on April 1, 2022.
CEO Tim Cook announced a 5% growth in their services segment, contributing $19 billion to the revenue.
In addition, Apple’s cash reserves were reported to be approximately $75 billion.
"""
 
# 3. Process the document with the NLP model
doc = nlp(financial_text)
 
# 4. Extract Named Entities (e.g., company names, monetary values, percentages)
entities = [(ent.text, ent.label_) for ent in doc.ents]
 
# 5. Display the extracted entities
print("Extracted Entities:\n")
for entity in entities:
    print(f"{entity[0]} ({entity[1]})")
 
# 6. Extract key financial figures using regular expressions (e.g., revenue, dividend, etc.)
# Here we are looking for currency values, percentages, and financial terms
revenue = re.findall(r"\$\d+(\.\d+)?\s?billion", financial_text)
dividends = re.findall(r"\$\d+\.\d+\s?per\s?share", financial_text)
 
print("\nExtracted Financial Information:\n")
print(f"Revenue: {revenue}")
print(f"Dividends: {dividends}")
 
# 7. Visualize the frequency of entities (such as company names or financial terms)
# Count frequency of terms (e.g., "Apple" or other company names)
company_names = [ent.text for ent in doc.ents if ent.label_ == "ORG"]
company_count = Counter(company_names)
 
print("\nCompany Name Frequency:\n")
for company, count in company_count.items():
    print(f"{company}: {count}")