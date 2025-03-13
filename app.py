import streamlit as st
import numpy as np
from PIL import Image

# Streamlit UI
st.title("🌿 Plant Disease Detection App (Without Model)")
st.sidebar.write("Upload an image to detect plant diseases.")

# Create three tabs
tab1, tab2, tab3 = st.tabs(["🍅 Tomato Disease", "🌶️ Pepper Disease", "🥔 Potato Disease"])

# Function to analyze leaf color and predict disease
def analyze_leaf(image):
    image = image.resize((256, 256))  # Resize for consistency
    image_np = np.array(image)  # Convert to numpy array
    
    # Compute average RGB values
    avg_color = image_np.mean(axis=(0, 1))  # Get mean for each channel

    # Basic rule-based classification
    if avg_color[1] > avg_color[0] and avg_color[1] > avg_color[2]:
        return "Healthy 🍃 (Mostly Green Leaf)"
    elif avg_color[0] > avg_color[1] and avg_color[0] > avg_color[2]:
        return "Early Blight 🍂 (Yellow/Brown Patches Detected)"
    elif avg_color[2] > avg_color[0] and avg_color[2] > avg_color[1]:
        return "Late Blight 🖤 (Dark Spots Detected)"
    else:
        return "Uncertain 🤔 (Upload a clearer leaf image)"

# Loop through each tab
for tab, disease in zip([tab1, tab2, tab3], ["Tomato Disease", "Pepper Disease", "Potato Disease"]):
    with tab:
        st.header(f"{disease}")
        uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"], key=disease)

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)

            # Get prediction using rule-based method
            predicted_class = analyze_leaf(image)
            st.success(f"🔍 **Prediction:** {predicted_class}")
