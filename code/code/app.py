import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image, ImageTk
import customtkinter as ctk
from tkinter import filedialog

# Load PyTorch Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PotatoCNN(nn.Module):
    def __init__(self, num_classes=3):
        super(PotatoCNN, self).__init__()
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

# Load the trained model
model = PotatoCNN(num_classes=3)
model.load_state_dict(torch.load("potato-detection-model.pth", map_location=device))
model.to(device)
model.eval()
print("Model loaded successfully!")

# Define Image Transformations
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# Define Class Names (Make sure they match your dataset)
class_names = [ "Potato___Early_blight","Potato___Late_blight","Potato___healthy"]

# Function to Predict Image Class
def predict_image(image_path):
    try:
        image = Image.open(image_path).convert("RGB")
        image = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(image)
            _, predicted = torch.max(output, 1)

        return class_names[predicted.item()]
    
    except Exception as e:
        print(f"Error processing image: {e}")
        return "Error"

# GUI Application using CustomTkinter
class DiseaseDetectionApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Plant Disease Detection")
        self.geometry("600x500")
        self.configure(bg="#2E2E2E")

        # Title Label
        self.label_title = ctk.CTkLabel(self, text="Plant Disease Detection", font=("Arial", 22, "bold"))
        self.label_title.pack(pady=10)

        # Upload Button
        self.btn_upload = ctk.CTkButton(self, text="Upload Image", command=self.upload_image, font=("Arial", 16))
        self.btn_upload.pack(pady=10)

        # Image Label
        self.image_label = ctk.CTkLabel(self, text="No Image Selected", font=("Arial", 14))
        self.image_label.pack(pady=10)

        # Display Image
        self.canvas = ctk.CTkCanvas(self, width=256, height=256, bg="white")
        self.canvas.pack(pady=10)

        # Prediction Label
        self.prediction_label = ctk.CTkLabel(self, text="Prediction: -", font=("Arial", 16, "bold"))
        self.prediction_label.pack(pady=10)

    def upload_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")])
        if file_path:
            self.display_image(file_path)
            predicted_class = predict_image(file_path)
            self.prediction_label.configure(text=f"Prediction: {predicted_class}")

    def display_image(self, image_path):
        image = Image.open(image_path)
        image = image.resize((256, 256))
        img_tk = ImageTk.PhotoImage(image)
        
        self.canvas.create_image(128, 128, image=img_tk)
        self.canvas.image = img_tk  # Prevent garbage collection

# Run the application
if __name__ == "__main__":
    app = DiseaseDetectionApp()
    app.mainloop()
