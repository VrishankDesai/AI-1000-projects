# Install if not already: pip install spacy
# Then download SpaCy model: python -m spacy download en_core_web_sm
 
import spacy
import re
 
# Load SpaCy's English model
nlp = spacy.load("en_core_web_sm")
 
# Sample user input for a flight booking bot
user_input = "I want to book a flight from New York to London on July 5th."
 
# Process the input
doc = nlp(user_input)
 
# Extract named entities as slots
slots = {
    "departure_city": None,
    "destination_city": None,
    "date": None
}
 
# Use SpaCy entities to fill 'date'
for ent in doc.ents:
    if ent.label_ == "GPE":
        if not slots["departure_city"]:
            slots["departure_city"] = ent.text
        elif not slots["destination_city"]:
            slots["destination_city"] = ent.text
    elif ent.label_ in ["DATE"]:
        slots["date"] = ent.text
 
# Output extracted slots
print("🧠 Slot Filling Results:\n")
for key, value in slots.items():
    print(f"🔹 {key}: {value if value else 'Not found'}")