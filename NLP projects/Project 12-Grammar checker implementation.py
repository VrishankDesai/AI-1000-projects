# Install if not already: pip install language-tool-python
 
import language_tool_python
 
# Initialize the grammar checking tool (English)
tool = language_tool_python.LanguageTool('en-US')
 
# Example sentences with grammar issues
sentences = [
    "She go to the store every day.",
    "I has a apple in my bag.",
    "Their going too the movies tonite.",
    "This are bad example of english.",
]
 
print("🧠 Grammar Checker Results:\n")
 
for sentence in sentences:
    # Check for grammar errors
    matches = tool.check(sentence)
    corrected = language_tool_python.utils.correct(sentence, matches)
    
    print(f"❌ Original:  {sentence}")
    print(f"✅ Corrected: {corrected}")
    print(f"🔧 Issues found: {len(matches)}\n")