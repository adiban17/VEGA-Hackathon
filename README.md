# VEGA: AI-Powered Medical Diagnostic System

VEGA is a comprehensive healthcare diagnostic platform developed for the **VEGA Hackathon**. It leverages deep learning models to assist in the detection of various medical conditions, including Alzheimer's, Brain Tumors, Lung Cancer, and Bone Fractures, through advanced medical imaging analysis.

## 🚀 Features
* **Multi-Disease Diagnostics**: A unified interface to detect multiple conditions using specialized AI architectures.
* **Alzheimer’s Detection**: Utilizes a DenseNet-based model for accurate classification.
* **Brain Tumor Identification**: Implements EfficientNet for high-accuracy tumor detection.
* **Lung Cancer Screening**: Features an EfficientNet-CBAM architecture for enhanced feature extraction from scans.
* **Fracture Analysis**: Employs DenseNet for detecting bone fractures in X-ray images.
* **Dual Interface**: Supports both a **Streamlit**-based web dashboard for quick analysis and a high-performance **FastAPI** backend for production deployment.

## 🛠️ Tech Stack

* **Frontend**: Streamlit
* **Backend**: FastAPI, Uvicorn
* **Deep Learning**: PyTorch, TensorFlow/Keras
* **Image Processing**: PIL (Pillow), NumPy
* **Data Visualization**: Matplotlib, Pandas

## 📁 Project Structure

```text
VEGA-Hackathon/
├── app.py                      # Streamlit Frontend Dashboard
├── backend/
│   ├── app.py                  # FastAPI Backend API
│   ├── alzheimers_densenet.keras
│   ├── brain_tumor_Efficient.pth
│   ├── fracture_densenet.keras
│   └── lungcancer_effifentnetcbam.pth
├── Alzheimers.ipynb            # Training Notebook: Alzheimer's
├── BrainTumor_EfficientNet.ipynb # Training Notebook: Brain Tumor
├── LungCancer_EfficentNet.ipynb  # Training Notebook: Lung Cancer
└── X_Ray.ipynb                 # Training Notebook: Fracture Detection
