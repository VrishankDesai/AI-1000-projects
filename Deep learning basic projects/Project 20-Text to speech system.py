# Install if not already: pip install pyttsx3
 
import pyttsx3
 
# Initialize the TTS engine
engine = pyttsx3.init()
 
# Set properties (optional)
engine.setProperty('rate', 150)       # Speed of speech
engine.setProperty('volume', 1.0)     # Volume (0.0 to 1.0)
 
# Choose a voice (optional)
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[0].id)  # Change index for different voices
 
# Input text
text = "Hello! I am your AI assistant. How can I help you today?"
 
# Speak the text
engine.say(text)
engine.runAndWait()