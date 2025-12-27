import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import os

# Load PyTorch Models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define CNN Model Class
class PlantDiseaseCNN(nn.Module):
    def __init__(self, num_classes):
        super(PlantDiseaseCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 32 * 32, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Load models dynamically
models_info = {
    "Tomato Disease Detection": {"model_path": "saved-model/saved-model/tomato-detection-model.pth", "dataset_path": "proper-dataset/tomato-detection"},
    "Pepper Disease Detection": {"model_path": "saved-model/saved-model/pepper-detection-model.pth", "dataset_path": "proper-dataset/peper-detection"},
    "Potato Disease Detection": {"model_path": "saved-model/saved-model/potato-detection-model.pth", "dataset_path": "proper-dataset/dataset-potato"},
}

models = {}
class_names = {}

for disease, info in models_info.items():
    # Load model
    state_dict = torch.load(info["model_path"], map_location=device)
    num_classes = state_dict['fc2.weight'].shape[0]  # Extract number of classes
    model = PlantDiseaseCNN(num_classes=num_classes)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    models[disease] = model

    # Extract class names from dataset folder
    dataset_path = info["dataset_path"]
    class_names[disease] = sorted([folder for folder in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, folder))])

# Define image transformation
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# Function to predict disease
def predict_image(image, model, disease):
    image = image.convert("RGB")
    image = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)

    return class_names[disease][predicted.item()]

# Streamlit UI
st.title("🌿 Plant Disease Detection App")
st.sidebar.write("Upload an image to detect plant diseases.")

# Create three tabs
tab1, tab2, tab3 = st.tabs(["🍅 Tomato Disease", "🌶️ Pepper Disease", "🥔 Potato Disease"])

for tab, disease in zip([tab1, tab2, tab3], models_info.keys()):
    with tab:
        st.header(f"{disease}")
        uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"], key=disease)

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)

            # Get prediction
            predicted_class = predict_image(image, models[disease], disease)
            st.success(f"🔍 **Prediction:** {predicted_class}")
