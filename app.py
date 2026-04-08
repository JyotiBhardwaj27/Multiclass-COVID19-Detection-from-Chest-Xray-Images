# import streamlit as st
# import torch
# import torch.nn as nn
# import cv2
# import numpy as np
# from PIL import Image

# # ================================
# # 1. DEVICE
# # ================================
# device = torch.device("cpu")

# # ================================
# # 2. MODEL
# # ================================
# class SimpleCNN(nn.Module):
#     def __init__(self, num_classes):
#         super(SimpleCNN, self).__init__()

#         self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
#         self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
#         self.conv3 = nn.Conv2d(64, 128, 3, padding=1)

#         self.pool = nn.MaxPool2d(2, 2)

#         self.fc1 = nn.Linear(128 * 28 * 28, 256)
#         self.fc2 = nn.Linear(256, num_classes)

#     def forward(self, x):
#         x = self.pool(torch.relu(self.conv1(x)))
#         x = self.pool(torch.relu(self.conv2(x)))
#         x = self.pool(torch.relu(self.conv3(x)))

#         x = x.view(x.size(0), -1)

#         x = torch.relu(self.fc1(x))
#         x = self.fc2(x)

#         return x

# # ================================
# # 3. LOAD MODEL
# # ================================
# model = SimpleCNN(3)
# model.load_state_dict(torch.load("artifacts/cnn_model.pth", map_location=device))
# model.eval()

# classes = ["Covid", "Normal", "Pneumonia"]

# # ================================
# # 4. GRAD-CAM
# # ================================
# class GradCAM:
#     def __init__(self, model, target_layer):
#         self.model = model
#         self.target_layer = target_layer

#         self.gradients = None
#         self.activations = None

#         self.target_layer.register_forward_hook(self.save_activation)
#         self.target_layer.register_backward_hook(self.save_gradient)

#     def save_activation(self, module, input, output):
#         self.activations = output

#     def save_gradient(self, module, grad_input, grad_output):
#         self.gradients = grad_output[0]

#     def generate(self, input_tensor, class_idx):
#         self.model.zero_grad()

#         output = self.model(input_tensor)
#         loss = output[:, class_idx]
#         loss.backward()

#         gradients = self.gradients
#         activations = self.activations

#         weights = torch.mean(gradients, dim=(2, 3), keepdim=True)

#         cam = torch.sum(weights * activations, dim=1).squeeze()
#         cam = torch.relu(cam)

#         cam = cam.detach().cpu().numpy()
#         cam = cv2.resize(cam, (224, 224))

#         cam = cam - cam.min()
#         cam = cam / cam.max()

#         return cam

# gradcam = GradCAM(model, model.conv3)

# # ================================
# # 5. PREPROCESS
# # ================================
# def preprocess(image):
#     img = np.array(image.convert("RGB"))
#     img = cv2.resize(img, (224, 224))

#     img = img / 255.0

#     # 🔥 ADD THIS (VERY IMPORTANT)
#     img = (img - 0.5) / 0.5

#     img = np.transpose(img, (2, 0, 1))
#     img = torch.tensor(img, dtype=torch.float).unsqueeze(0)

#     return img
# # ================================
# # 6. UI
# # ================================
# st.title("🩺 COVID-19 Detection from Chest X-rays")
# st.write("Upload an X-ray image to get prediction and Grad-CAM visualization")

# uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

# if uploaded_file is not None:
#     image = Image.open(uploaded_file)

#     st.image(image, caption="Uploaded Image", width="stretch")

#     input_tensor = preprocess(image)

#     output = model(input_tensor)
#     pred_class = torch.argmax(output).item()

#     st.success(f"Prediction: {classes[pred_class]}")

#     # Grad-CAM
#     cam = gradcam.generate(input_tensor, pred_class)

#     original = np.array(image.convert("RGB"))
#     original = cv2.resize(original, (224, 224))

#     heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
#     overlay = cv2.addWeighted(original, 0.6, heatmap, 0.4, 0)

#     st.subheader("Grad-CAM Visualization")

#     col1, col2 = st.columns(2)

#     with col1:
#         st.image(original, caption="Original", width="stretch")

#     with col2:
#         st.image(overlay, caption="Grad-CAM",width="stretch")

import streamlit as st
import torch
import torch.nn as nn
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import os

# ================================
# PAGE CONFIG
# ================================
st.set_page_config(page_title="Chest X-ray AI", layout="wide")

# ================================
# DEVICE
# ================================
device = torch.device("cpu")

# ================================
# MODEL
# ================================
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)

        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(128 * 28 * 28, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))

        x = x.view(x.size(0), -1)

        x = torch.relu(self.fc1(x))
        x = self.fc2(x)

        return x

# ================================
# LOAD MODEL (ROBUST PATH)
# ================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "artifacts", "cnn_model.pth")

model = SimpleCNN(3)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

classes = ["Covid", "Normal", "Pneumonia"]

# ================================
# GRAD-CAM
# ================================
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer

        self.gradients = None
        self.activations = None

        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate(self, input_tensor, class_idx):
        self.model.zero_grad()

        output = self.model(input_tensor)
        loss = output[:, class_idx]
        loss.backward()

        gradients = self.gradients
        activations = self.activations

        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)

        cam = torch.sum(weights * activations, dim=1).squeeze()
        cam = torch.relu(cam)

        cam = cam.detach().cpu().numpy()
        cam = cv2.resize(cam, (224, 224))

        cam = cam - cam.min()
        cam = cam / cam.max()

        return cam

gradcam = GradCAM(model, model.conv3)

# ================================
# PREPROCESS
# ================================
def preprocess(image):
    img = np.array(image.convert("RGB"))
    img = cv2.resize(img, (224, 224))
    img = img / 255.0

    # IMPORTANT normalization
    img = (img - 0.5) / 0.5

    img = np.transpose(img, (2, 0, 1))
    img = torch.tensor(img, dtype=torch.float).unsqueeze(0)

    return img

# ================================
# UI
# ================================
st.title("🩺 AI-Based Chest X-ray Diagnosis")
st.write("Upload an X-ray image to detect COVID-19, Normal, or Pneumonia with explainability")

uploaded_file = st.file_uploader("📤 Upload X-ray Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    st.image(image, caption="Uploaded Image", width="stretch")

    input_tensor = preprocess(image)

    # Prediction
    output = model(input_tensor)
    probs = torch.softmax(output, dim=1)
    pred_class = torch.argmax(probs).item()
    confidence = probs[0][pred_class].item()

    # Display prediction
    st.markdown(f"## 🧾 Prediction: **{classes[pred_class]}**")
    st.info(f"Confidence: {confidence:.2f}")

    # ================================
    # PROBABILITY CHART
    # ================================
    prob_values = probs.detach().numpy()[0]

    df = pd.DataFrame({
        "Class": classes,
        "Probability": prob_values
    })

    st.subheader("📊 Class Probabilities")
    st.bar_chart(df.set_index("Class"))

    # ================================
    # MEDICAL INTERPRETATION
    # ================================
    st.subheader("🧠 Medical Insight")

    if pred_class == 0:
        st.warning("COVID detected: Diffuse lung involvement patterns observed.")
    elif pred_class == 1:
        st.success("Normal lungs: No significant abnormalities detected.")
    else:
        st.error("Pneumonia detected: Localized infection patterns observed.")

    # ================================
    # GRAD-CAM
    # ================================
    cam = gradcam.generate(input_tensor, pred_class)

    original = np.array(image.convert("RGB"))
    original = cv2.resize(original, (224, 224))

    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(original, 0.6, heatmap, 0.4, 0)

    st.subheader("🔥 Grad-CAM Visualization")

    col1, col2 = st.columns(2)

    with col1:
        st.image(original, caption="Original", width="stretch")

    with col2:
        st.image(overlay, caption="Grad-CAM", width="stretch")

# ================================
# EXTRA INFO
# ================================
with st.expander("ℹ️ How does this work?"):
    st.write("""
    This application uses a Convolutional Neural Network (CNN) trained on chest X-ray images.
    Grad-CAM highlights the regions of the lungs that influenced the model’s decision.
    """)

st.caption("⚠️ This tool is for educational purposes only and not for medical diagnosis.")