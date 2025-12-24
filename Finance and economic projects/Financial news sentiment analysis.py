from textblob import TextBlob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
 
# 1. Simulate a dataset of financial news headlines
data = {
    "headline": [
        "Apple hits record high in stock price",
        "Apple faces regulatory challenges in Europe",
        "Apple launches new iPhone model",
        "Apple's revenue growth slows in Q4",
        "Apple announces plans for environmental sustainability"
    ]
}
 
df = pd.DataFrame(data)
 
# 2. Define a function to calculate sentiment polarity
def get_sentiment(text):
    blob = TextBlob(text)
    return blob.sentiment.polarity
 
# 3. Apply sentiment analysis to the headlines
df['sentiment'] = df['headline'].apply(get_sentiment)
 
# 4. Classify sentiment into categories
def classify_sentiment(polarity):
    if polarity > 0:
        return "Positive"
    elif polarity < 0:
        return "Negative"
    else:
        return "Neutral"
 
df['sentiment_class'] = df['sentiment'].apply(classify_sentiment)
 
# 5. Display the results
print("Financial News Sentiment Analysis Results:\n")
print(df)
 
# 6. Plot the sentiment distribution
sentiment_counts = df['sentiment_class'].value_counts()
 
plt.figure(figsize=(8, 6))
sentiment_counts.plot(kind='bar', color=['green', 'red', 'blue'])
plt.title("Sentiment Distribution of Financial News Headlines")
plt.xlabel("Sentiment")
plt.ylabel("Count")
plt.xticks(rotation=0)
plt.show()