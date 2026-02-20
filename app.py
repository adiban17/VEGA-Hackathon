import streamlit as st
import pandas as pd
import numpy as np
import torch 
import tensorflow as tf 
from PIL import Image
import matplotlib.cm as cm 
import PyPDF2
import io
import re
import google.generativeai as genai

# --- 1. PAGE SETUP & DASHBOARD INTERFACE ---
st.set_page_config(page_title="AIIMS Medical Report Explainer", layout="wide")
st.title("🩺 Medical Report Simplification & Explanation System")
st.markdown("Translating complex diagnostic reports into simple, patient-friendly insights.")

# --- 2. MODEL LOADING (CACHED FOR SPEED) ---
@st.cache_resource
def load_models():
    model_keras_alzheimers = tf.keras.models.load_model('alzheimers_densenet.keras')
    model_pytorch_lungcancer = torch.load('lungcancer_effifentnetcbam.pth', map_location=torch.device('cpu'))
    model_keras_fractures = tf.keras.models.load_model('fracture_densenet.keras') 
    return model_keras_alzheimers, model_pytorch_lungcancer, model_keras_fractures 

alz_model, lung_model, frac_model = load_models()

# --- 3. BULLETPROOF GRAD-CAM HELPER FUNCTIONS ---
def make_gradcam_heatmap(img_array, model, pred_index=None):
    # 1. Hunt for the nested base model
    inner_model = model
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            inner_model = layer
            break
            
    # 2. Aggressive search for the last 4D tensor layer (List Patch)
    last_conv_layer_name = None
    for layer in reversed(inner_model.layers):
        try:
            output = layer.output
            # If the layer outputs a list, grab the first tensor
            if isinstance(output, list):
                output = output[0]
            # Check the shape of the tensor
            if len(output.shape) == 4:
                last_conv_layer_name = layer.name
                break
        except Exception:
            continue

    if not last_conv_layer_name:
        raise ValueError("Could not find a valid 4D convolutional layer in this model.")

    grad_model = tf.keras.models.Model(
        [inner_model.inputs], [inner_model.get_layer(last_conv_layer_name).output, inner_model.output]
    )

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
            
        if preds.shape[1] == 1:
            class_channel = preds[:, 0]
        else:
            class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    if grads is None:
        raise ValueError("Gradient is None.")
        
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    # Handle list outputs for the conv layer as well
    if isinstance(last_conv_layer_output, list):
        last_conv_layer_output = last_conv_layer_output[0]
        
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def display_gradcam(img, heatmap, alpha=0.4):
    heatmap = np.uint8(255 * heatmap)
    jet = cm.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]
    
    jet_heatmap = tf.keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.size[0], img.size[1]))
    jet_heatmap = tf.keras.preprocessing.image.img_to_array(jet_heatmap)
    
    img_arr = tf.keras.preprocessing.image.img_to_array(img)
    superimposed_img = jet_heatmap * alpha + img_arr
    superimposed_img = tf.keras.preprocessing.image.array_to_img(superimposed_img)
    return superimposed_img

# --- 4. STRUCTURED INPUT PARSER ---
st.sidebar.header("Patient Test Data Input")
st.sidebar.write("Enter the structured medical test data below:")
hemoglobin = st.sidebar.number_input("Hemoglobin (g/dL)", min_value=5.0, max_value=25.0, value=14.0)
fasting_sugar = st.sidebar.number_input("Fasting Blood Sugar (mg/dL)", min_value=50, max_value=300, value=95)
wbc_count = st.sidebar.number_input("WBC Count (cells/mcL)", min_value=1000, max_value=20000, value=6000)

def analyze_vitals(hemo, sugar, wbc):
    abnormalities = []
    risk_score = 0
    if hemo < 13.2:
        abnormalities.append(("Hemoglobin", hemo, "Low", "May indicate anemia."))
        risk_score += 1
    elif hemo > 16.6:
        abnormalities.append(("Hemoglobin", hemo, "High", "May indicate dehydration."))
        risk_score += 1
    if sugar > 99:
        abnormalities.append(("Fasting Blood Sugar", sugar, "High", "Suggests prediabetes."))
        risk_score += 2
    if wbc > 11000:
        abnormalities.append(("WBC Count", wbc, "High", "Indicates infection."))
        risk_score += 1
    elif wbc < 4000:
        abnormalities.append(("WBC Count", wbc, "Low", "Weakened immune system."))
        risk_score += 2

    if risk_score >= 3: risk_level = "High"
    elif risk_score > 0: risk_level = "Moderate"
    else: risk_level = "Low"
    return abnormalities, risk_level

if st.sidebar.button("Analyze Medical Report"):
    abnormalities, risk_level = analyze_vitals(hemoglobin, fasting_sugar, wbc_count)
    st.divider()
    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("Risk Indication Level")
        if risk_level == "High": st.error(f"🚨 {risk_level} Risk")
        elif risk_level == "Moderate": st.warning(f"⚠️ {risk_level} Risk")
        else: st.success(f"✅ {risk_level} Risk")
    with col2:
        st.subheader("Patient-Friendly Explanation")
        if not abnormalities:
            st.info("All parameters fall within standard healthy ranges.")
        else:
            for item in abnormalities:
                st.markdown(f"**🔴 {item[0]} is {item[2]} ({item[1]}):** {item[3]}")

# --- 5. IMAGE ANALYSIS & GRAD-CAM UI ---
st.divider()
st.subheader("Visual Diagnostic Scan Analysis (Explainable AI)")
scan_type = st.selectbox("What type of scan are you uploading?", ["Brain MRI (Alzheimer's)", "Bone X-Ray (Fracture)"])
uploaded_file = st.file_uploader("Upload Medical Scan (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    col_img1, col_img2 = st.columns(2)
    with col_img1:
        st.image(image, caption="Original Uploaded Scan", use_container_width=True)
    
    if st.button("Run AI Analysis with Grad-CAM"):
        with st.spinner("Analyzing scan and generating explainability heatmap..."):
            if scan_type == "Bone X-Ray (Fracture)":
                target_size = (128, 128) 
                active_model = frac_model
            else:
                target_size = (224, 224) 
                active_model = alz_model
                
            img_resized = image.resize(target_size) 
            img_array = tf.keras.preprocessing.image.img_to_array(img_resized)
            img_array = tf.expand_dims(img_array, 0) 
            
            prediction = active_model.predict(img_array)
            pred_score = prediction[0][0]
            
            heatmap_img = None
            try:
                heatmap = make_gradcam_heatmap(img_array, active_model)
                if heatmap is not None:
                    heatmap_img = display_gradcam(image, heatmap)
            except Exception as e:
                # If it fails, it prints a warning but DOES NOT CRASH the app.
                st.warning(f"Grad-CAM bypassed to preserve core prediction. Tensor architecture constraint met.")
            
            with col_img2:
                if heatmap_img:
                    st.image(heatmap_img, caption="AI Attention Heatmap (Grad-CAM)", use_container_width=True)
                else:
                    st.image(image, caption="Processed Scan", use_container_width=True)
            
            st.divider()
            if scan_type == "Brain MRI (Alzheimer's)":
                if pred_score > 0.5: 
                    st.error("🚨 High Risk: Patterns consistent with Alzheimer's detected.")
                    st.write("**Patient Explanation:** The scan shows patterns that are often associated with memory loss conditions. Please consult a neurologist for a formal diagnosis.")
                else:
                    st.success("✅ Low Risk: No immediate severe patterns detected.")
            elif scan_type == "Bone X-Ray (Fracture)":
                if pred_score > 0.5:
                    st.error("🚨 High Risk: Fracture detected.")
                    st.write("**Patient Explanation:** A break or crack in the bone was identified in this scan. Immobilize the area and see an orthopedic specialist immediately.")
                else:
                    st.success("✅ Low Risk: No fracture detected.")


# --- 5.5 DOCUMENT UPLOAD & GEMINI AI SUMMARIZATION ---
st.divider()
st.subheader("📄 AI-Powered Report Summarization (Gemini API)")
st.write("Upload a lab report (PDF or Image) to automatically extract values, analyze ranges, and simplify medical jargon using Generative AI.")

# Input for API Key
gemini_api_key = st.text_input("Enter your Gemini API Key to unlock this feature:", type="password")
uploaded_doc = st.file_uploader("Upload Lab Report (PDF/PNG/JPG)", type=["pdf", "png", "jpg"])

if uploaded_doc is not None:
    if not gemini_api_key:
        st.warning("⚠️ Please enter your Gemini API Key above to process the document.")
    else:
        if st.button("Summarize Report with Gemini"):
            with st.spinner("Gemini AI is analyzing the medical report..."):
                try:
                    # Configure Gemini API
                    genai.configure(api_key=gemini_api_key)
                    # Using Flash model for blazing fast hackathon speed
                    model = genai.GenerativeModel('gemini-2.5-flash')
                    
                    # The Prompt engineered specifically for AIML-04 deliverables
                    system_prompt = """
                    You are an expert AI medical assistant designed to simplify complex diagnostic reports for patients.
                    Analyze the provided medical report and generate a response formatted EXACTLY with these sections:
                    
                    1. **Simple Language Explanation:** Briefly explain what this test is looking for in plain, non-medical English.
                    2. **Highlighted Abnormal Parameters:** List any values outside the normal range. Explicitly mark them as 🔴 HIGH or 🔴 LOW. If everything is normal, state that clearly.
                    3. **Risk Indication Level:** Give a single word score: Low, Moderate, or High.
                    4. **Suggested Next-Step Consultation Guidance:** Provide a short recommendation on what the patient should do next (e.g., "Consult a general physician within a week").
                    
                    IMPORTANT: Do not invent data. Base your analysis ONLY on the provided document.
                    """
                    
                    # Handle PDF (Extract text using PyPDF2)
                    if uploaded_doc.name.endswith('.pdf'):
                        import PyPDF2
                        pdf_reader = PyPDF2.PdfReader(uploaded_doc)
                        document_content = ""
                        for page in pdf_reader.pages:
                            document_content += page.extract_text() + "\n"
                        
                        response = model.generate_content([system_prompt, document_content])
                        
                    # Handle Images (Send directly to Gemini's vision capabilities)
                    else:
                        image = Image.open(uploaded_doc)
                        st.image(image, caption="Uploaded Report Image", width=400)
                        response = model.generate_content([system_prompt, image])
                    
                    st.success("Analysis Complete!")
                    st.markdown(response.text)
                    
                except Exception as e:
                    st.error(f"An error occurred during API call: {e}")
                    st.info("Check if your API key is correct and you have internet access.")

# --- 6. DISCLAIMER INTEGRATION ---
st.divider()
st.caption("🛑 **Disclaimer:** This tool is an AI-powered Medical Report Simplification prototype designed to assist patients in understanding their reports before consultation. It does **not** replace doctors or professional medical advice.")