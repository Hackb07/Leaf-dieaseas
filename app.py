import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import os

# --------------------------------------------------
# Device Configuration
# --------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------------------------------------
# CNN Model Definition
# --------------------------------------------------
class PlantDiseaseCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(64 * 32 * 32, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        return self.fc2(x)

# --------------------------------------------------
# Model & Dataset Configuration
# --------------------------------------------------
models_info = {
    "Tomato Disease": {
        "model_path": "saved-model/saved-model/tomato-detection-model.pth",
        "dataset_path": "proper-dataset/tomato-detection"
    },
    "Pepper Disease": {
        "model_path": "saved-model/saved-model/pepper-detection-model.pth",
        "dataset_path": "proper-dataset/peper-detection"
    },
    "Potato Disease": {
        "model_path": "saved-model/saved-model/potato-detection-model.pth",
        "dataset_path": "proper-dataset/dataset-potato"
    }
}

models = {}
class_names = {}

# --------------------------------------------------
# Load Models & Class Names
# --------------------------------------------------
for disease, info in models_info.items():
    state_dict = torch.load(info["model_path"], map_location=device)

    num_classes = state_dict["fc2.weight"].shape[0]
    model = PlantDiseaseCNN(num_classes)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    models[disease] = model

    # Load class names from dataset folders
    class_names[disease] = sorted([
        d for d in os.listdir(info["dataset_path"])
        if os.path.isdir(os.path.join(info["dataset_path"], d))
    ])

# --------------------------------------------------
# Image Transform
# --------------------------------------------------
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                         std=[0.5, 0.5, 0.5])
])

# --------------------------------------------------
# Prediction Function
# --------------------------------------------------
def predict_image(image, model, disease):
    image = image.convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        pred_index = torch.argmax(outputs, dim=1).item()

    return class_names[disease][pred_index]

# --------------------------------------------------
# Streamlit UI
# --------------------------------------------------
st.title("🌿 Plant Disease Detection App")
st.sidebar.info("Upload a leaf image to detect plant disease")

tab1, tab2, tab3 = st.tabs(["🍅 Tomato", "🌶️ Pepper", "🥔 Potato"])

for tab, disease in zip([tab1, tab2, tab3], models_info.keys()):
    with tab:
        st.subheader(disease)
        uploaded_file = st.file_uploader(
            "Upload leaf image",
            type=["jpg", "jpeg", "png"],
            key=disease
        )

        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)

            prediction = predict_image(image, models[disease], disease)

            st.success(f"🧠 **Predicted Class:** {prediction}")
