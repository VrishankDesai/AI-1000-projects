# Install dependencies:
# pip install transformers torch torchvision pillow
 
from transformers import BlipProcessor, BlipForQuestionAnswering
from PIL import Image
import torch
 
# Load model and processor
processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")
model.eval()
 
# Load and display the image
image_path = "zebra.jpg"  # Replace with your image
image = Image.open(image_path).convert('RGB')
 
# Define the question
question = "What animal is in the picture?"
 
# Preprocess inputs
inputs = processor(image, question, return_tensors="pt")
 
# Inference
with torch.no_grad():
    out = model.generate(**inputs)
 
# Decode and print answer
answer = processor.decode(out[0], skip_special_tokens=True)
print("🖼️ Question:", question)
print("🧠 Answer:", answer)