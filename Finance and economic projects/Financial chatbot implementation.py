import random
import re
 
# 1. Define a simple rule-based financial chatbot
class FinancialChatbot:
    def __init__(self):
        self.responses = {
            "greet": ["Hello! How can I assist you with your finances today?", "Hi there! What financial questions do you have?"],
            "account_balance": ["Your current account balance is $5,200.", "You have $5,200 available in your account."],
            "loan_status": ["Your loan application is under review and will be processed in 3-5 business days.",
                             "We are currently reviewing your loan application, and you will hear back within 3-5 days."],
            "investment_advice": ["I recommend diversifying your portfolio with a mix of stocks, bonds, and ETFs.",
                                  "You should consider investing in low-cost index funds for long-term growth."],
            "default": ["I'm sorry, I didn't quite catch that. Can you ask me something else related to finance?"]
        }
 
    def process_query(self, query):
        query = query.lower()
 
        # Check for keywords in the query to respond appropriately
        if re.search(r"(hello|hi|hey)", query):
            return random.choice(self.responses["greet"])
        elif re.search(r"(account balance|balance)", query):
            return random.choice(self.responses["account_balance"])
        elif re.search(r"(loan status|loan application)", query):
            return random.choice(self.responses["loan_status"])
        elif re.search(r"(investment advice|investing)", query):
            return random.choice(self.responses["investment_advice"])
        else:
            return self.responses["default"]
 
# 2. Create the chatbot
chatbot = FinancialChatbot()
 
# 3. Simulate a conversation with the chatbot
print("Chatbot: Hi, I'm your financial assistant. How can I help you today?")
while True:
    user_input = input("You: ")
    if user_input.lower() in ['exit', 'quit', 'bye']:
        print("Chatbot: Goodbye! Have a great day.")
        break
    response = chatbot.process_query(user_input)
    print("Chatbot:", response)