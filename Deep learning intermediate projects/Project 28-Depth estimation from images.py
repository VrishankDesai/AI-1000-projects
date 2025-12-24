# Install dependencies:
# pip install torch torchvision opencv-python matplotlib
 
import torch
import torchvision.transforms as T
import cv2
import matplotlib.pyplot as plt
import numpy as np
 
# Load MiDaS model from PyTorch Hub
model_type = "DPT_Large"  # Alternatives: DPT_Hybrid, MiDaS_small
midas = torch.hub.load("intel-isl/MiDaS", model_type)
midas.eval()
 
# Move to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
midas.to(device)
 
# Load transforms
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = midas_transforms.dpt_transform if "DPT" in model_type else midas_transforms.small_transform
 
# Load and preprocess image
img_path = "room.jpg"  # Replace with your image path
img = cv2.imread(img_path)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
input_tensor = transform(img_rgb).to(device)
 
# Inference
with torch.no_grad():
    prediction = midas(input_tensor.unsqueeze(0))
    prediction = torch.nn.functional.interpolate(
        prediction.unsqueeze(1),
        size=img.shape[:2],
        mode="bicubic",
        align_corners=False,
    ).squeeze()
 
depth_map = prediction.cpu().numpy()
 
# Normalize depth map for visualization
depth_vis = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
 
# Plot results
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Input Image")
plt.imshow(img_rgb)
plt.axis("off")
 
plt.subplot(1, 2, 2)
plt.title("Predicted Depth Map")
plt.imshow(depth_vis, cmap='inferno')
plt.axis("off")
plt.show()