
# 🚦 Smart Traffic AI — Helmet Detection System

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-purple)
![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-red?logo=streamlit)
![mAP50](https://img.shields.io/badge/mAP50-96.1%25-brightgreen)
![License](https://img.shields.io/badge/License-MIT-yellow)

A real-time **helmet violation detection system** built with a dual YOLOv8 model pipeline and an EasyOCR-powered license plate reader. Deployed as an interactive Streamlit web application designed for automated traffic safety enforcement.

> 🎓 Computer Vision Semester Project — COMSATS University Islamabad | BS Artificial Intelligence (2024–2028)

---

## 🎯 Problem Statement

Motorcycle helmet non-compliance is a leading cause of fatal road accidents in Pakistan and globally. Manual monitoring at traffic checkpoints is impractical at scale. This system automates detection using computer vision — identifying riders, bikes, and helmet status in real time from images or video streams.

---

## ✨ Key Features

- 🔍 **Dual-model YOLO pipeline** — scene-level detection (bikes & riders) + helmet classification
- 🪖 **Two-class detection** — `With Helmet` / `Without Helmet` with color-coded overlays
- 🔢 **EasyOCR license plate reading** — extracts plate numbers from violation frames
- 📋 **Automated violation logging** — saves timestamped CSV logs of all detected violations
- 🖥️ **Streamlit web interface** — supports image and video input with live annotations


## 📊 Model Performance

| Metric | With Helmet | Without Helmet | Overall |
|--------|-------------|----------------|---------|
| **mAP50** | 0.955 | 0.967 | **0.961** |
| mAP50-95 | 0.775 | 0.738 | 0.757 |
| Precision | 0.955 | 0.907 | 0.931 |
| Recall | 0.936 | 0.950 | 0.943 |

> 🚀 Inference speed: **2.8ms/image** (~350 FPS) — suitable for real-time video processing.


## 🏗️ Project Structure


smart-traffic-ai-helmet-detection/
│
├── ui/
│   ├── app.py          # Main Streamlit application
│   ├── logger.py       # Violation CSV logging
│   └── overlay.py      # Bounding box drawing and label rendering
│
├── models/
│   ├── best.pt         # Custom-trained YOLOv8 (helmet classification)
│   └── yolov8n.pt      # Pre-trained YOLOv8n — COCO (bike & rider detection)
│
├── logs/               # Auto-generated violation logs (CSV)
├── requirements.txt
└── README.md

## ⚙️ How It Works

The pipeline runs two YOLO models in sequence on each frame:

| Step | Action | Output |
|------|--------|--------|
| 1 | Load image or video frame | Raw numpy array |
| 2 | YOLOv8n (COCO): detect bikes & riders | Bounding boxes + class IDs |
| 3 | YOLOv8n (custom): classify helmet status | `With Helmet` / `Without Helmet` + confidence |
| 4 | Draw color-coded annotations | Orange=bike, Yellow=rider, Green=helmet, Red=violation |
| 5 | EasyOCR: extract license plate text | Plate number string |
| 6 | Log violation + export annotated output | CSV log + result image/video |

---

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- pip

### Installation


# 1. Clone the repository
git clone https://github.com/hasana157/smart-traffic-ai-helmet-detection.git
cd smart-traffic-ai-helmet-detection

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the Streamlit app
streamlit run ui/app.py

The app will open in your browser at `http://localhost:8501`.


## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| [YOLOv8](https://github.com/ultralytics/ultralytics) | Object detection & classification |
| [EasyOCR](https://github.com/JaidedAI/EasyOCR) | License plate text extraction |
| [Streamlit](https://streamlit.io/) | Web application interface |
| [OpenCV](https://opencv.org/) | Video frame processing |
| [Roboflow](https://roboflow.com/) | Dataset (1,376 images, 2 classes) |

---

## 📦 Dataset

- **Source:** Bike Helmet Detection — Roboflow (2024)
- **Size:** 1,376 images | 80% train / 10% val / 10% test
- **Classes:** `With Helmet`, `Without Helmet`
- **Augmentations:** Horizontal flip, rotation ±30°, shear, blur, mosaic

---

## 👩‍💻 Author

**Hasana Zahid**
BS Artificial Intelligence, COMSATS University Islamabad  
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?logo=linkedin)](https://www.linkedin.com/in/hasana-zahid-605543310)
[![GitHub](https://img.shields.io/badge/GitHub-Profile-black?logo=github)](https://github.com/hasana157)
```

---
