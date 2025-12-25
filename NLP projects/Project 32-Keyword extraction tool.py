# Install if not already: pip install scikit-learn nltk
 
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import stopwords
import numpy as np
 
nltk.download('stopwords')
 
# Sample text for keyword extraction
text = """
Artificial intelligence (AI) is revolutionizing many industries, including healthcare, finance, and transportation. 
Through machine learning, natural language processing, and computer vision, AI systems are automating complex tasks 
and enabling intelligent decision-making in real time.
"""
 
# Define TF-IDF Vectorizer
vectorizer = TfidfVectorizer(stop_words=stopwords.words('english'))
tfidf_matrix = vectorizer.fit_transform([text])
 
# Get feature names and scores
feature_array = np.array(vectorizer.get_feature_names_out())
tfidf_scores = tfidf_matrix.toarray().flatten()
 
# Rank top N keywords
N = 10
top_indices = tfidf_scores.argsort()[-N:][::-1]
top_keywords = feature_array[top_indices]
 
# Output top keywords
print("🧠 Top Keywords Extracted:\n")
for i, keyword in enumerate(top_keywords, 1):
    print(f"{i}. {keyword}")