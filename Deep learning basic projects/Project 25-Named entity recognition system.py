# Install if not already: pip install spacy
# Download the English model: python -m spacy download en_core_web_sm
 
import spacy
from spacy import displacy
 
# Load spaCy English model
nlp = spacy.load("en_core_web_sm")
 
# Sample text
text = "Barack Obama was born in Hawaii and served as the 44th President of the United States. He studied at Harvard University."
 
# Process text
doc = nlp(text)
 
# Print named entities
print("Named Entities:\n")
for ent in doc.ents:
    print(f"{ent.text:<30} ➤ {ent.label_}")
 
# Optional: Visualize entities in a browser
displacy.render(doc, style="ent", jupyter=False)