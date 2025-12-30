import pandas as pd
from textblob import TextBlob
import matplotlib.pyplot as plt
 
# Simulated brand-related tweets or comments
comments = [
    "I love this brand! The new collection is amazing.",
    "Terrible customer service. Will not buy again.",
    "Product is okay, not too great, not too bad.",
    "Excellent experience with support team!",
    "The quality has really gone down recently.",
    "Fast delivery and great packaging.",
    "Not worth the money at all.",
    "Their app is super smooth and easy to use.",
    "Pretty average, nothing special.",
    "Worst return process I've ever experienced."
]
 
# Create DataFrame
df = pd.DataFrame({'Comment': comments})
 
# Classify sentiment using polarity score
def get_sentiment(text):
    polarity = TextBlob(text).sentiment.polarity
    if polarity > 0.2:
        return 'Positive'
    elif polarity < -0.2:
        return 'Negative'
    else:
        return 'Neutral'
 
# Apply sentiment classifier
df['Sentiment'] = df['Comment'].apply(get_sentiment)
 
# Show sentiment analysis results
print("Brand Sentiment Analysis:")
print(df)
 
# Visualize sentiment distribution
plt.figure(figsize=(6, 4))
df['Sentiment'].value_counts().plot(kind='bar', color='coral')
plt.title('Brand Sentiment Distribution')
plt.xlabel('Sentiment')
plt.ylabel('Number of Mentions')
plt.grid(axis='y')
plt.tight_layout()
plt.show()