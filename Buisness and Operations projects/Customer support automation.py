import re
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
 
# Predefined FAQ responses
faq_responses = {
    'order status': "You can check your order status by logging into your account and visiting 'My Orders'.",
    'return policy': "Our return policy allows returns within 30 days of delivery.",
    'cancel order': "To cancel an order, go to 'My Orders', select the order, and click 'Cancel'.",
    'payment failed': "Please check your payment method or try using a different card.",
    'delivery time': "Delivery usually takes 3–5 business days depending on your location."
}
 
# Lowercase keywords for matching
keywords = list(faq_responses.keys())
 
# Function to respond to user queries
def get_support_response(user_input):
    # Preprocess and tokenize user input
    user_input = user_input.lower()
    tokens = word_tokenize(user_input)
 
    # Check for matching keyword in user input
    for key in keywords:
        key_tokens = word_tokenize(key)
        if any(token in tokens for token in key_tokens):
            return faq_responses[key]
    return "I'm sorry, I didn't understand that. Could you please rephrase your question?"
 
# Test the bot
queries = [
    "How do I cancel my order?",
    "Tell me about your return policy.",
    "When will my delivery arrive?",
    "My payment failed. What should I do?",
    "What's the status of my order?"
]
 
for q in queries:
    print(f"User: {q}")
    print(f"Bot: {get_support_response(q)}\n")