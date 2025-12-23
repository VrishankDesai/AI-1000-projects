# Install if not already: pip install SpeechRecognition pyaudio
 
import speech_recognition as sr
 
# Initialize recognizer
recognizer = sr.Recognizer()
 
# Use microphone as input
with sr.Microphone() as source:
    print("🎤 Say something! I'm listening...")
    recognizer.adjust_for_ambient_noise(source)
    audio = recognizer.listen(source)
 
# Recognize speech using Google Web Speech API
try:
    text = recognizer.recognize_google(audio)
    print("📝 You said:", text)
except sr.UnknownValueError:
    print("❌ Could not understand audio")
except sr.RequestError as e:
    print(f"❌ Could not request results; {e}")
