import re

#A dictionary that maps keywords to predefined responses
responses = {
    "hello": "Hi there! How can I assist you today?",
    "hi": "Hello! What can I do for you?",
    "how are you": "I'm just a bot, but thanks for asking! How can I help you?",
    "what is your name": "I'm ChatBot, your virtual assistant.",
    "help": "Sure! What do you need help with?",
    "bye": "Goodbye! Have a great day!",
    "thanks you": "You're welcome! Happy to help!",
    "default": "I'm not sure how to respond to that. Could you rephrase?"
}

#Function to find appropriate response based on user input
def chatbot_response(user_input):
    # Convert the input to lowercase
    user_input = user_input.lower()
    
    # Check for keywords in the user input
    for keyword in responses.keys():
        if re.search(keyword, user_input):
            return responses[keyword]
    
    # Return default response if no keywords matched
    return responses["default"]

#Main function to run chatbot
def chatbot():
    print("ChatBot: Hello i am here to assist you. (type bye to exit)")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            print("ChatBot: Goodbye! Have a great day!")
            break
        
        # Get the chatbot response based on user input
        response = chatbot_response(user_input)
        print(f"ChatBot: {response}")
        
#Run the chatbot
chatbot()