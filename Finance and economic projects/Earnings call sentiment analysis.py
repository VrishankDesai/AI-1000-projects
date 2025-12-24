import pandas as pd
import numpy as np
from textblob import TextBlob
import re
import matplotlib.pyplot as plt
 
# 1. Simulate earnings call transcript
earnings_call_transcript = """
CEO: Good morning, everyone. We're excited to report that for Q1, the company has achieved a revenue growth of 12% year-over-year.
We have successfully launched our new product line, which has exceeded initial projections by 20%.
Our operating income has grown by 15%, and we are on track to meet our 2022 targets.
CFO: The company has maintained strong liquidity with over $2 billion in cash reserves. Our expenses were well-controlled, and margins improved.
We are investing heavily in R&D and new product development. Our team remains focused on driving sustainable growth.
CEO: We're optimistic about the future, and we believe that our recent acquisitions will position us for even greater success in the coming quarters.
"""
 
# 2. Perform sentiment analysis on the transcript
def get_sentiment(text):
    blob = TextBlob(text)
    return blob.sentiment.polarity
 
# Apply sentiment analysis to the full transcript
overall_sentiment = get_sentiment(earnings_call_transcript)
 
# 3. Extract key financial terms using regex
financial_terms = re.findall(r"\$[0-9,]+(?:\.[0-9]{1,2})?", earnings_call_transcript)  # Extract monetary values
growth_terms = re.findall(r"(revenue growth|operating income|net income|margins|cash reserves)", earnings_call_transcript, re.IGNORECASE)
 
# 4. Visualize sentiment over time (optional)
# For this, we could break the transcript into sections and assign sentiment to each. For simplicity, we're using the overall sentiment here.
sentiment_label = "Positive" if overall_sentiment > 0 else "Negative" if overall_sentiment < 0 else "Neutral"
 
# 5. Display the results
print("Earnings Call Sentiment Analysis and Key Insights Extraction:\n")
print(f"Overall Sentiment: {sentiment_label} (Polarity: {overall_sentiment:.2f})")
print("\nKey Financial Terms Found in Transcript:")
print(financial_terms)
print("\nKey Growth Metrics Found:")
print(growth_terms)
 
# 6. Plot sentiment (in this simple case, we plot the overall sentiment)
plt.figure(figsize=(8, 6))
plt.bar([sentiment_label], [overall_sentiment], color='green' if overall_sentiment > 0 else 'red')
plt.title("Earnings Call Sentiment")
plt.xlabel("Sentiment")
plt.ylabel("Sentiment Score")
plt.show()