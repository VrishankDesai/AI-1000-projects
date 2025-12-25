# Install dependencies:
# pip install spacy coreferee
# python -m spacy download en_core_web_trf
# python -m coreferee install en
 
import spacy
import coreferee
 
# Load SpaCy transformer-based English model with coreferee
nlp = spacy.load("en_core_web_trf")
nlp.add_pipe('coreferee')
 
# Sample text with pronouns and repeated entities
text = """
John is a software engineer. He works at Google. 
Every day, John rides his bike to the office. His colleagues admire him for his dedication.
"""
 
# Process the text
doc = nlp(text)
 
# Display coreference chains
print("🧠 Coreference Resolution Results:\n")
 
if doc._.has_coref:
    for chain in doc._.coref_chains:
        print("🔗 Coreference Chain:")
        for mention in chain:
            print(f" - {mention}")
        print()
else:
    print("⚠️ No coreferences found.")