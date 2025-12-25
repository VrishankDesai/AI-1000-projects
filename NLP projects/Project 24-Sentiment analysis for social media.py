# Install if not already: pip install nltk
 
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
 
nltk.download('vader_lexicon')
 
# Initialize VADER sentiment analyzer
sia = SentimentIntensityAnalyzer()
 
# Example social media posts
tweets = [
    "I'm so happy with the new update! 🚀🔥",
    "This is the worst experience I've ever had. #fail",
    "Meh, it's okay I guess. Nothing special.",
    "Loving the new features! Great job @devteam!",
    "Why is this so broken again? I'm tired of this..."
]
 
print("🧠 Sentiment Analysis on Social Media Posts:\n")
 
for tweet in tweets:
    scores = sia.polarity_scores(tweet)
    sentiment = "positive" if scores['compound'] > 0.05 else "negative" if scores['compound'] < -0.05 else "neutral"
    
    print(f"💬 Tweet: {tweet}")
    print(f"🔎 Sentiment: {sentiment} (Score: {scores['compound']})\n")