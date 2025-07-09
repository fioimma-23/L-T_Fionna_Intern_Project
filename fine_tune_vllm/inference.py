import streamlit as st

st.set_page_config(page_title="L&T Building Classifier", layout="centered")

import torch
from torchvision import models, transforms
from torch import nn
from PIL import Image
import requests

# -----------------------
# Load Model and Classes
# -----------------------

@st.cache_resource
def load_model():
    model = models.mobilenet_v3_large()
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, 4)
    # model.load_state_dict(torch.load("building_classifier.pt", map_location=torch.device("cpu")))
    checkpoint = torch.load("building_class.pt", map_location=torch.device("cpu"))
    model.load_state_dict(checkpoint['state_dict'])
    class_names = checkpoint['class_names']
    normalization_stats = checkpoint['normalization']
    model.eval()
    return model, class_names, normalization_stats

model, class_names, normalization_stats = load_model()
class_names = ['CRR Centre', 'EDRC', 'Technology Centre-2', 'Technology Centre-3']

# -----------------------
# Transforms
# -----------------------

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225])
])

# -----------------------
# Building Metadata
# -----------------------

building_info = {
    "CRR Centre": {
        "name": "Administrative Office Building (CRR Centre)",
        "alias": "CRR Centre",
        "location": "Chennai - L&T Campus",
        "awards": [
            "1994 Federation Internationale du Beton (fib) Award"
        ],
        "structure": "Square structure lifted above ground on 4 core shafts with deep well foundations. Overlapping 21.6m square superstructures with post-tensioned pyramid bases.",
        "features": [
            "Inverted four-sided hollow pyramid base cantilevered 10m",
            "Precast waffle slabs for flexible floor design",
            "Entrance unobstructed and open"
        ],
        "green_features": [],
        "notable": "Floating design architecture with massive 5000t load-bearing core pillars"
    },
    "EDRC": {
        "name": "Engineering Design and Research Centre (EDRC)",
        "alias": "EDRC",
        "location": "Chennai - L&T Campus",
        "awards": [
            "2002 Federation Internationale du Beton (fib) Award",
            "LEED Silver Rating - USGBC (Existing Building)"
        ],
        "structure": "Triangular design with two symmetrical wings on either side of a service core. Built-up area of 8686 sq.m (G+4 floors)",
        "features": [
            "Terraced, landscaped sky gardens",
            "Natural daylight access to floor interiors",
            "Architectural form blending with structure"
        ],
        "green_features": [
            "LEED-EB Silver Certified Green Building"
        ],
        "notable": "Titled 'Tree of Knowledge'"
    },  
    "Technology Centre-2": {
        "name": "Technology Centre II (TCII)",
        "alias": "Technology Centre-2",
        "location": "Chennai - L&T Campus",
        "awards": [
            "LEED-NC v2.1 Certified Green Building"
        ],
        "structure": "Expanding floor plate with central service core, column-free large office spaces, built on 1.5 lakh sq.ft",
        "features": [
            "Post-tensioned EPS embedded flat slabs",
            "RC core + shear wall system on pile foundation",
            "648 TR centralized air-conditioning with air-cooled screw chillers",
            "100% power back-up",
            "Automatic addressable fire detection system"
        ],
        "green_features": [
            "Roof with high albedo (U-value: 0.22 Btu/hr-sqft°F)",
            "Double glazing & over-deck insulation",
            "Energy-efficient lighting",
            "Recycled water for irrigation",
            "Non-CFC chillers",
            "Automated flow/flush fixtures",
            "Building automation for energy and water",
            "Storm water control",
            "Water metering"
        ],
        "notable": "Flexible to adapt functional changes for 50+ years"
    },
    "Technology Centre-3": {
        "name": "Technology Centre III (TCIII)",
        "alias": "Technology Centre-3",
        "location": "Chennai - L&T Campus, rear end",
        "awards": [
            "LEED Silver Rating - IGBC (New Construction)",
            "Ultratech Award for Outstanding Concrete Structure of Tamil Nadu (2011)"
        ],
        "structure": "Twin tower (8 floors) with 3-level basement; 8.31 lakh sq.ft total area; 5.5 lakh office space + 2.81 lakh basement",
        "features": [
            "Parking for 703 cars, 932 two-wheelers",
            "Skywalk and top-floor corporate suites",
            "Canteen, training, business, recreation at ground level"
        ],
        "green_features": [
            "Sensor-controlled water fixtures",
            "Pervious pavers for storm water",
            "100% covered & reserved parking for carpools",
            "High performance glazing",
            "Ozone-free refrigerant chillers",
            "Water metering",
            "Outdoor & indoor air quality monitoring"
        ],
        "notable": "State-of-the-art green twin tower with skywalk"
    }
}

# -----------------------
# Helper Functions
# -----------------------

def predict_building(image, confidence_threshold=0.7):
    image_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        max_prob, pred_idx = torch.max(probabilities, 1)

    confidence = max_prob.item()

    if confidence < confidence_threshold:
        return "unknown", confidence
    else:
        return class_names[pred_idx.item()], confidence
    
        # if confidence.item() < 0.7:
        #     return "unknown", confidence.item()
        # else:
        #     return class_names[pred.item()], confidence.item()
    #     _, pred = torch.max(outputs, 1)
    # return class_names[pred.item()]

def generate_prompt(building, user_question):
    info = building_info[building]
    return f"""
You are a helpful assistant answering queries about L&T buildings in Chennai. Answer the user's question using ONLY the facts below. Do NOT add any extra information or commentary.

Building: {building}
Alias: {info['alias']}
Location: {info['location']}
Structure: {info['structure']}
Awards: {info['awards']}
Features:
{chr(10).join('- ' + feature for feature in info['features'])}
Notable: {info['notable']}

Answer the following user question based on the building info:
{user_question}
### Response Rules
1. Answer concisely in 1-3 sentences
2. Use ONLY the above facts
3. If information isn't in facts, say "I don't have that information"
4. Never ask follow-up questions
"""

def query_ollama(prompt):
    try:
        res = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "llama3",
                "prompt": prompt,
                "stream": False
            }
        )
        return res.json().get("response", "No response from model.")
    except Exception as e:
        return f"Error calling Ollama: {str(e)}"

# -----------------------
# Streamlit UI
# -----------------------

st.title("L&T Building Identifier + Assistant")

uploaded_image = st.file_uploader("Upload a photo of the building", type=["jpg", "jpeg", "png"])

if uploaded_image:
    image = Image.open(uploaded_image).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    predicted_building, confidence = predict_building(image)
    if predicted_building == "unknown":
        st.warning(f"This building is unknown (confidence: {confidence:.2f})")
    else:
        st.success(f"This looks like: **{predicted_building}** (confidence: {confidence:.2f})")

    question = st.text_input("Ask a question about this building:")
    if question:
        prompt = generate_prompt(predicted_building, question)
        with st.spinner("Thinking with LLaMA..."):
            response = query_ollama(prompt)
        st.markdown("**LLaMA's Response:**")
        st.write(response)
