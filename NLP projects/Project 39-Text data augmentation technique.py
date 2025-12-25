# Install if not already: pip install nltk
 
import random
import nltk
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
 
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
 
# Sample sentence
sentence = "Data augmentation is useful for improving model performance."
 
# Synonym Replacement
def synonym_replacement(text, n=2):
    words = word_tokenize(text)
    new_words = words.copy()
    count = 0
 
    for i, word in enumerate(words):
        synonyms = set()
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                synonym = lemma.name().replace("_", " ").lower()
                if synonym != word.lower():
                    synonyms.add(synonym)
 
        if synonyms:
            synonym = random.choice(list(synonyms))
            new_words[i] = synonym
            count += 1
        if count >= n:
            break
 
    return " ".join(new_words)
 
# Random Insertion
def random_insertion(text, n=2):
    words = word_tokenize(text)
    for _ in range(n):
        word = random.choice(words)
        synonyms = wordnet.synsets(word)
        if synonyms:
            synonym_list = [lemma.name().replace("_", " ") for lemma in synonyms[0].lemmas()]
            if synonym_list:
                insert_word = random.choice(synonym_list)
                insert_pos = random.randint(0, len(words))
                words.insert(insert_pos, insert_word)
    return " ".join(words)
 
# Display Augmented Variants
print("🧠 Original Sentence:\n", sentence)
 
aug1 = synonym_replacement(sentence, n=2)
print("\n🔁 Synonym Replacement:\n", aug1)
 
aug2 = random_insertion(sentence, n=2)
print("\n🧩 Random Insertion:\n", aug2)