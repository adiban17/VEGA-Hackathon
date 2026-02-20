from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import tensorflow as tf
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import io
import base64
import matplotlib.cm as cm
import google.generativeai as genai
import PyPDF2

# --- 1. INITIALIZE API & CORS ---
app = FastAPI(title="AIIMS Medical Report API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 2. LOAD MODELS ---
print("Loading AI Models into memory...")

alz_model = tf.keras.models.load_model('alzheimers_densenet.keras')
frac_model = tf.keras.models.load_model('fracture_densenet.keras')

def load_efficientnet_state_dict(filepath):
    """Dynamically reads the number of classes and loads the weights"""
    state_dict = torch.load(filepath, map_location=torch.device('cpu'))
    
    # Auto-detect number of classes from the checkpoint's final layer weight shape
    if 'classifier.1.weight' in state_dict:
        num_classes = state_dict['classifier.1.weight'].shape[0]
    else:
        num_classes = 2 # Fallback
        
    print(f"Loaded {filepath} with {num_classes} classes.")
    
    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model

brain_tumor_model = load_efficientnet_state_dict('brain_tumor_Efficient.pth')
lung_cancer_model = load_efficientnet_state_dict('lungcancer_effifentnetcbam.pth')

print("All Keras and PyTorch Models successfully loaded!")

# --- 3. HELPER FUNCTIONS ---
def encode_image_to_base64(img):
    buffered = io.BytesIO()
    img.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def make_gradcam_heatmap(img_array, model, pred_index=None):
    inner_model = model
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            inner_model = layer
            break
            
    last_conv_layer_name = None
    for layer in reversed(inner_model.layers):
        try:
            output = layer.output
            if isinstance(output, list):
                output = output[0]
            if len(output.shape) == 4:
                last_conv_layer_name = layer.name
                break
        except Exception:
            continue

    if not last_conv_layer_name:
        raise ValueError("Could not find a valid 4D convolutional layer.")

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

# --- 4. API ENDPOINTS ---
class VitalsInput(BaseModel):
    hemoglobin: float
    fasting_sugar: float
    wbc_count: float

@app.post("/api/analyze-vitals")
def analyze_vitals_endpoint(data: VitalsInput):
    abnormalities = []
    risk_score = 0
    
    if data.hemoglobin < 13.2:
        abnormalities.append({"parameter": "Hemoglobin", "value": data.hemoglobin, "status": "Low", "explanation": "May indicate anemia."})
        risk_score += 1
    elif data.hemoglobin > 16.6:
        abnormalities.append({"parameter": "Hemoglobin", "value": data.hemoglobin, "status": "High", "explanation": "May indicate dehydration."})
        risk_score += 1
        
    if data.fasting_sugar > 99:
        abnormalities.append({"parameter": "Fasting Blood Sugar", "value": data.fasting_sugar, "status": "High", "explanation": "Levels above 99 mg/dL suggest prediabetes."})
        risk_score += 2
        
    if data.wbc_count > 11000:
        abnormalities.append({"parameter": "WBC Count", "value": data.wbc_count, "status": "High", "explanation": "Often indicates an infection or inflammation."})
        risk_score += 1
    elif data.wbc_count < 4000:
        abnormalities.append({"parameter": "WBC Count", "value": data.wbc_count, "status": "Low", "explanation": "Can indicate a weakened immune system."})
        risk_score += 2

    if risk_score >= 3:
        risk_level = "High"
    elif risk_score > 0:
        risk_level = "Moderate"
    else:
        risk_level = "Low"
        
    return {
        "status": "success",
        "risk_level": risk_level,
        "abnormalities": abnormalities
    }

@app.post("/api/analyze-scan")
async def analyze_scan_endpoint(scan_type: str = Form(...), file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        heatmap_base64 = None
        heatmap_generated = False
        is_high_risk = False
        
        # --- PYTORCH INFERENCE BLOCK ---
        if scan_type in ["Brain MRI (Tumor)", "Chest CT (Lung Cancer)"]:
            active_model = brain_tumor_model if "Tumor" in scan_type else lung_cancer_model
            
            preprocess = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            input_tensor = preprocess(image).unsqueeze(0) 
            
            with torch.no_grad():
                output = active_model(input_tensor)
                
                if output.numel() == 1:
                    pred_score = torch.sigmoid(output).item()
                    is_high_risk = pred_score > 0.5
                else:
                    # Multi-class logic (4 classes)
                    probabilities = torch.softmax(output, dim=1)[0]
                    pred_class_idx = torch.argmax(probabilities).item()
                    
                    # Assuming class '2' is "Normal/No Tumor" in your dataset 
                    # (Standard for 4-class alphabetical Kaggle datasets: 0=Adeno/Glioma, 1=LargeCell/Meningioma, 2=Normal/NoTumor, 3=Squamous/Pituitary)
                    # If it says High Risk on healthy images, change `!= 2` to the correct normal index (0, 1, or 3).
                    normal_class_index = 2 
                    is_high_risk = (pred_class_idx != normal_class_index)
            
        # --- KERAS INFERENCE BLOCK ---
        else:
            active_model = frac_model if "Fracture" in scan_type else alz_model
            target_size = (128, 128) if "Fracture" in scan_type else (224, 224)
                
            img_resized = image.resize(target_size) 
            img_array = tf.keras.preprocessing.image.img_to_array(img_resized)
            img_array = tf.expand_dims(img_array, 0) 
            
            prediction = active_model.predict(img_array)
            pred_score = float(prediction[0][0])
            is_high_risk = pred_score > 0.5
            
            try:
                heatmap = make_gradcam_heatmap(img_array, active_model)
                if heatmap is not None:
                    heatmap_img = display_gradcam(image, heatmap)
                    heatmap_base64 = encode_image_to_base64(heatmap_img)
                    heatmap_generated = True
            except Exception as e:
                print(f"Grad-CAM bypassed: {e}")
        
        # --- DYNAMIC EXPLANATION GENERATOR ---
        if scan_type == "Brain MRI (Tumor)":
            title = "🚨 High Risk: Suspicious mass detected." if is_high_risk else "✅ Low Risk: No tumor patterns detected."
            explanation = "The scan highlights patterns consistent with a brain tumor. Consult an oncologist immediately." if is_high_risk else "The brain structures appear standard."
        elif scan_type == "Chest CT (Lung Cancer)":
            title = "🚨 High Risk: Pulmonary nodules detected." if is_high_risk else "✅ Low Risk: Lungs appear clear."
            explanation = "The AI detected patterns often associated with lung cancer. Please schedule a specialist consultation." if is_high_risk else "No high-risk nodules detected in the lung tissue."
        elif scan_type == "Brain MRI (Alzheimer's)":
            title = "🚨 High Risk: Patterns consistent with Alzheimer's." if is_high_risk else "✅ Low Risk: No immediate severe patterns detected."
            explanation = "Please consult a neurologist for a formal diagnosis." if is_high_risk else "Maintain routine checkups."
        else: 
            title = "🚨 High Risk: Fracture detected." if is_high_risk else "✅ Low Risk: No fracture detected."
            explanation = "Immobilize the area and see an orthopedic specialist immediately." if is_high_risk else "The bone structure appears intact."

        return {
            "status": "success",
            "risk_title": title,
            "patient_explanation": explanation,
            "heatmap_generated": heatmap_generated,
            "heatmap_base64": f"data:image/jpeg;base64,{heatmap_base64}" if heatmap_generated else None
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/summarize-report")
async def summarize_report_endpoint(api_key: str = Form(...), file: UploadFile = File(...)):
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        system_prompt = """
        You are an expert AI medical assistant designed to simplify complex diagnostic reports for patients.
        Analyze the provided medical report and generate a response formatted EXACTLY with these sections:
        
        1. **Simple Language Explanation:** Briefly explain what this test is looking for in plain, non-medical English.
        2. **Highlighted Abnormal Parameters:** List any values outside the normal range. Explicitly mark them as 🔴 HIGH or 🔴 LOW. If everything is normal, state that clearly.
        3. **Risk Indication Level:** Give a single word score: Low, Moderate, or High.
        4. **Suggested Next-Step Consultation Guidance:** Provide a short recommendation on what the patient should do next (e.g., "Consult a general physician within a week").
        """
        
        if file.filename.lower().endswith('.pdf'):
            pdf_bytes = await file.read()
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))
            document_content = ""
            for page in pdf_reader.pages:
                extracted = page.extract_text()
                if extracted:
                    document_content += extracted + "\n"
                
            response = model.generate_content([system_prompt, document_content])
            
        else:
            image_bytes = await file.read()
            image = Image.open(io.BytesIO(image_bytes))
            response = model.generate_content([system_prompt, image])
            
        return {"status": "success", "summary_markdown": response.text}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))