import pandas as pd
from textblob import TextBlob
from collections import Counter
import matplotlib.pyplot as plt
 
# Simulated product reviews
reviews = [
    "The camera quality is excellent, but the battery life is disappointing.",
    "I love the screen resolution and the design is sleek.",
    "Battery drains too fast. The software also feels sluggish.",
    "Great performance and build quality, but the screen scratches easily.",
    "Audio output is amazing, though the charging port feels loose.",
    "I’m happy with the processor speed, and the touch response is smooth.",
    "The camera captures vivid colors but struggles in low light.",
    "The phone overheats and the battery doesn’t last long.",
    "Beautiful design, responsive UI, but storage is too low.",
    "Display is crisp, but the device gets hot during gaming."
]
 
# Create a DataFrame
df = pd.DataFrame({'Review': reviews})
 
# Function to extract noun phrases and analyze sentiment
def extract_features_with_sentiment(text):
    blob = TextBlob(text)
    features = [phrase.lower() for phrase in blob.noun_phrases]
    sentiment = blob.sentiment.polarity
    return features, sentiment
 
# Analyze all reviews
feature_sentiment_data = []
 
for review in df['Review']:
    features, polarity = extract_features_with_sentiment(review)
    for feature in features:
        feature_sentiment_data.append({'Feature': feature, 'Sentiment': polarity})
 
# Convert to DataFrame
features_df = pd.DataFrame(feature_sentiment_data)
 
# Aggregate sentiment and frequency per feature
summary = features_df.groupby('Feature').agg(
    Frequency=('Sentiment', 'count'),
    AvgSentiment=('Sentiment', 'mean')
).sort_values(by='Frequency', ascending=False)
 
# Display most discussed features
print("Top Product Features Mentioned and Their Sentiment:")
print(summary.head(10))
 
# Visualize feature frequency and sentiment
plt.figure(figsize=(10, 5))
summary.head(10).plot(kind='barh', y='Frequency', legend=False, color='steelblue')
plt.title("Top 10 Most Frequently Mentioned Product Features")
plt.xlabel("Mention Frequency")
plt.ylabel("Feature")
plt.gca().invert_yaxis()
plt.grid(True)
plt.tight_layout()
plt.show()