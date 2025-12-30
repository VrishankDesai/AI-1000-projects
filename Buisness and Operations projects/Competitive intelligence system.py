import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt
 
# Simulated news and updates from competitor websites or press releases
documents = [
    "Competitor A launches AI-powered analytics dashboard.",
    "Competitor B announces price reduction on core SaaS product.",
    "Competitor A partners with cloud provider for global expansion.",
    "Competitor C introduces real-time customer support feature.",
    "Competitor B expands into European market with new tools.",
    "Competitor C raises funding to enhance AI capabilities."
]
 
# Create DataFrame
df = pd.DataFrame({'Update': documents})
 
# Extract keyword frequencies using CountVectorizer
vectorizer = CountVectorizer(stop_words='english')
X = vectorizer.fit_transform(df['Update'])
word_freq = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())
total_freq = word_freq.sum().sort_values(ascending=False)
 
# Display top keywords
print("Top Competitive Intelligence Keywords:")
print(total_freq.head(10))
 
# Generate word cloud for visual pattern recognition
wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(total_freq)
 
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Competitor Activity Word Cloud')
plt.tight_layout()
plt.show()