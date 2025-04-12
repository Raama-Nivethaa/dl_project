import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms
import os

# ✅ THIS MUST BE THE FIRST STREAMLIT COMMAND
st.set_page_config(page_title="Alzheimer’s Prediction", layout="centered")

# ========== Model Definition ==========

class SimpleANN(nn.Module):
    def __init__(self):
        super(SimpleANN, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(3 * 64 * 64, 512)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(512, 256)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(256, 4)

    def forward(self, x):
        x = self.flatten(x)
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        return self.fc3(x)

# ========== Configuration ==========

CLASS_NAMES = ['MildDemented', 'ModerateDemented', 'NonDemented', 'VeryMildDemented']
MODEL_PATH = r"C:\Users\neel\OneDrive\Desktop\dl_project\models\my_model.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========== Load Model ==========

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model file not found at: {MODEL_PATH}")
        st.stop()
    model = SimpleANN().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    return model

model = load_model()

# ========== Image Preprocessing ==========

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

def predict_image(image: Image.Image):
    img = transform(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        outputs = model(img)
        _, predicted = torch.max(outputs, 1)
    return CLASS_NAMES[predicted.item()]

# ========== Streamlit UI ==========

st.title("🧠 Alzheimer's Disease Prediction App")
st.write("Upload a brain MRI image to classify its Alzheimer's stage.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("Classifying...")
    prediction = predict_image(image)
    st.success(f"🧠 **Prediction:** {prediction}")
