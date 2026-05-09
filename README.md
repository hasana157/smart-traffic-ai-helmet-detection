# Smart Traffic AI - Helmet Detection System

This is a Streamlit-based web application that uses YOLO models for object detection (detecting riders, bikes, and helmets) and EasyOCR for reading number plates.

## Project Structure

```
final/
│
├── ui/
│   ├── app.py          # Main Streamlit application
│   ├── logger.py       # Handles logging violations
│   └── overlay.py      # Handles drawing bounding boxes and labels
│
├── models/
│   ├── best.pt         # Trained YOLO model for helmet detection
│   └── yolov8n.pt      # Pre-trained YOLO model for COCO (riders/bikes)
│
├── logs/               # Directory where violation CSV logs are saved
│
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

## How to Run

1. Open your terminal and navigate to the `final` folder.
   ```powershell
   cd "c:\Users\hasan\Downloads\Helmet Detection System\final"
   ```

2. Install the required dependencies:
   ```powershell
   pip install -r requirements.txt
   ```

3. Run the Streamlit application from the `ui` folder:
   ```powershell
   streamlit run ui/app.py
   ```
   *Alternatively, you can `cd ui` first and run `streamlit run app.py`.*

4. Your web browser should automatically open the app at `http://localhost:8501`.
