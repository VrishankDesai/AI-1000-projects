# Install if not already: pip install pandas scikit-learn
 
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
 
# Sample dataset of essays and human scores (mock data)
data = {
    "essay": [
        "The importance of education in modern society cannot be overstated...",
        "Dogs are loyal animals and great companions to humans...",
        "Climate change is a pressing global issue that needs immediate attention...",
        "Technology has drastically changed how we communicate in the 21st century...",
        "Sports help in the physical and mental development of students..."
    ],
    "score": [9, 6, 8, 7.5, 7]
}
 
df = pd.DataFrame(data)
 
# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(df["essay"], df["score"], test_size=0.3, random_state=42)
 
# Define TF-IDF + Linear Regression pipeline
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("regressor", LinearRegression())
])
 
# Train model
pipeline.fit(X_train, y_train)
 
# Predict and evaluate
predicted_scores = pipeline.predict(X_test)
mse = mean_squared_error(y_test, predicted_scores)
 
print("🧠 Automated Essay Scoring:\n")
for essay, actual, pred in zip(X_test, y_test, predicted_scores):
    print(f"📝 Essay Snippet: {essay[:60]}...")
    print(f"✅ Actual Score: {actual} | 🤖 Predicted Score: {pred:.2f}\n")
 
print(f"📉 Mean Squared Error of Model: {mse:.2f}")