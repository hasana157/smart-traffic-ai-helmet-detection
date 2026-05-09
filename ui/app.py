import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import torch
import easyocr
import cv2
import pandas as pd
import streamlit as st
import tempfile
import numpy as np

from ultralytics import YOLO
from overlay import draw_overlay
from logger import save_log, is_duplicate

# ---------------- PATHS ----------------
BASE_DIR          = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR         = os.path.join(BASE_DIR, "models")
LOG_DIR           = os.path.join(BASE_DIR, "logs")
YOLO_HELMET_PATH  = os.path.join(MODEL_DIR, "best.pt")       # Your trained model — helmet/no-helmet
YOLO_COCO_PATH    = os.path.join(MODEL_DIR, "yolov8n.pt")    # COCO model — rider/bike detection

# COCO class IDs
PERSON_CLASS     = 0
MOTORCYCLE_CLASS = 3

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Smart Traffic AI",
    layout="wide",
    page_icon="🚦",
    initial_sidebar_state="expanded"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@500;600;700&family=DM+Sans:wght@400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* ── Main area ── */
[data-testid="stAppViewContainer"],
[data-testid="stMain"] {
    background: #f8fafc;
    color: #0f172a;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #0f172a !important;
    border-right: 3px solid #2563eb;
}

[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] div,
[data-testid="stSidebar"] label {
    color: #f1f5f9 !important;
}

[data-testid="stSidebar"] .stMarkdown p {
    color: #94a3b8 !important;
    font-size: 0.8rem;
}

[data-testid="stSidebar"] .stButton > button {
    background: #1e40af !important;
    color: #ffffff !important;
    border: 1px solid #3b82f6 !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
}

[data-testid="stSidebar"] .stButton > button:hover {
    background: #2563eb !important;
}

/* ── Sidebar title ── */
.main-title {
    font-family: 'Rajdhani', sans-serif;
    font-size: 2.1rem;
    font-weight: 700;
    color: #ffffff;
    letter-spacing: 2px;
    text-transform: uppercase;
    line-height: 1.1;
    margin-bottom: 0;
}

.subtitle {
    font-size: 0.72rem;
    color: #60a5fa;
    letter-spacing: 3px;
    text-transform: uppercase;
    margin-top: 4px;
    margin-bottom: 1.2rem;
    font-weight: 500;
}

/* ── Stat cards ── */
.stat-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 0.85rem 1.1rem;
    margin-bottom: 0.55rem;
    position: relative;
    overflow: hidden;
    box-shadow: 0 2px 6px rgba(0,0,0,0.06);
}

.stat-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0;
    width: 5px; height: 100%;
}

.stat-card.green::before  { background: #16a34a; }
.stat-card.red::before    { background: #dc2626; }
.stat-card.yellow::before { background: #b45309; }
.stat-card.blue::before   { background: #2563eb; }

.stat-label {
    font-size: 0.67rem;
    color: #334155;
    letter-spacing: 2px;
    text-transform: uppercase;
    font-weight: 700;
    margin-bottom: 2px;
}

.stat-value {
    font-family: 'Rajdhani', sans-serif;
    font-size: 2.1rem;
    font-weight: 700;
    line-height: 1;
}

.stat-value.green  { color: #15803d; }
.stat-value.red    { color: #b91c1c; }
.stat-value.yellow { color: #92400e; }
.stat-value.blue   { color: #1d4ed8; }

/* ── Section headers ── */
.section-header {
    font-family: 'Rajdhani', sans-serif;
    font-size: 0.72rem;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: #1e293b;
    border-bottom: 2px solid #cbd5e1;
    padding-bottom: 5px;
    margin-bottom: 10px;
    margin-top: 14px;
    font-weight: 700;
}

/* ── Upload hint ── */
.upload-hint {
    text-align: center;
    color: #334155;
    font-size: 0.88rem;
    padding: 2.5rem 1rem;
    border: 2px dashed #93c5fd;
    border-radius: 12px;
    margin-top: 1rem;
    background: #eff6ff;
    font-weight: 500;
}

/* ── Dataframe table ── */
[data-testid="stDataFrame"] {
    border-radius: 8px;
    overflow: hidden;
    border: 1px solid #cbd5e1 !important;
}

[data-testid="stDataFrame"] th {
    background-color: #1e3a8a !important;
    color: #ffffff !important;
    font-size: 0.68rem !important;
    letter-spacing: 1.5px !important;
    text-transform: uppercase !important;
    font-weight: 700 !important;
    padding: 8px 10px !important;
}

[data-testid="stDataFrame"] td {
    color: #0f172a !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    background: #ffffff !important;
    border-bottom: 1px solid #e2e8f0 !important;
}

[data-testid="stDataFrame"] tr:nth-child(even) td { background: #f1f5f9 !important; }
[data-testid="stDataFrame"] tr:hover td           { background: #dbeafe !important; }

/* ── General headings and text ── */
h1, h2, h3, h4 { color: #0f172a !important; font-weight: 700 !important; }
p, li, span     { color: #1e293b; }

/* ── Streamlit alerts ── */
[data-testid="stAlert"] { border-radius: 8px; }

/* ── Hide default Streamlit chrome ── */
#MainMenu, footer { visibility: hidden; }
div[data-testid="stStatusWidget"] { display: none; }

/* ── File uploader: bright blue dropzone ── */
[data-testid="stFileUploaderDropzone"] {
    background: #1e3a8a !important;
    border: 2px dashed #60a5fa !important;
    border-radius: 10px !important;
    padding: 0.8rem !important;
}

[data-testid="stFileUploaderDropzone"] button {
    background: #2563eb !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 700 !important;
    font-size: 0.85rem !important;
    padding: 0.45rem 1.1rem !important;
    transition: background 0.2s ease !important;
}

[data-testid="stFileUploaderDropzone"] button:hover {
    background: #3b82f6 !important;
    cursor: pointer !important;
}

[data-testid="stFileUploaderDropzone"] span,
[data-testid="stFileUploaderDropzone"] small,
[data-testid="stFileUploaderDropzone"] p {
    color: #bfdbfe !important;
    font-size: 0.75rem !important;
}
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODELS ----------------
@st.cache_resource
def load_models():
    # COCO model — detects riders (person) and bikes (motorcycle)
    bike_model   = YOLO(YOLO_COCO_PATH)
    # Your trained model — detects helmet / no-helmet
    helmet_model = YOLO(YOLO_HELMET_PATH)
    return bike_model, helmet_model

@st.cache_resource
def load_ocr():
    return easyocr.Reader(['en'])

bike_model, helmet_ai = load_models()
ocr_reader = load_ocr()

# ---------------- STATE ----------------
if "violations" not in st.session_state:
    st.session_state.violations = []
if "plates_detected" not in st.session_state:
    st.session_state.plates_detected = []
if "total_riders" not in st.session_state:
    st.session_state.total_riders = 0
if "frames_processed" not in st.session_state:
    st.session_state.frames_processed = 0

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.markdown('<div class="main-title">🚦 Traffic<br>AI</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Violation Detection System</div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Upload Image or Video",
        type=["jpg", "jpeg", "png", "mp4", "avi", "mov"],
        help="Supports JPG, PNG images and MP4, AVI, MOV videos"
    )

    st.markdown('<div class="section-header">Detection Settings</div>', unsafe_allow_html=True)
    conf_threshold   = st.slider("Confidence Threshold", 0.1, 1.0, 0.4, 0.05)
    show_conf_scores = st.checkbox("Show Confidence Scores", value=True)
    process_every_n  = st.slider("Process Every N Frames (video)", 1, 5, 1,
                                 help="Skip frames for faster processing")

    st.markdown('<div class="section-header">Session</div>', unsafe_allow_html=True)
    if st.button("🗑️ Clear All Records", use_container_width=True):
        st.session_state.violations      = []
        st.session_state.plates_detected = []
        st.session_state.total_riders    = 0
        st.session_state.frames_processed = 0
        st.rerun()

    st.markdown('<div class="section-header">Legend</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:0.78rem; color:#64748b; line-height:2">
    🟢 <b style="color:#4ade80">Green box</b> — Helmet detected<br>
    🔴 <b style="color:#f87171">Red box</b> — No helmet (violation)<br>
    🟡 <b style="color:#facc15">Yellow box</b> — Number plate<br>
    🟠 <b style="color:#fb923c">Orange box</b> — Motorcycle<br>
    🔵 <b style="color:#60a5fa">Blue box</b> — Rider/Person
    </div>
    """, unsafe_allow_html=True)

# ---------------- MAIN LAYOUT ----------------
col_feed, col_panel = st.columns([3, 1.1])

with col_feed:
    st.markdown('<div class="section-header">Live Detection Feed</div>', unsafe_allow_html=True)
    st_frame = st.empty()

    if not uploaded_file:
        st.markdown("""
        <div class="upload-hint">
            📁 Upload an image or video from the sidebar to begin detection
        </div>
        """, unsafe_allow_html=True)

with col_panel:
    st.markdown('<div class="section-header">Session Stats</div>', unsafe_allow_html=True)

    stats_riders     = st.empty()
    stats_violations = st.empty()
    stats_plates     = st.empty()
    stats_frames     = st.empty()

    def render_stats():
        total_v = len(st.session_state.violations)
        total_p = len(st.session_state.plates_detected)
        riders  = st.session_state.total_riders
        frames  = st.session_state.frames_processed

        stats_riders.markdown(f"""
        <div class="stat-card blue">
            <div class="stat-label">Riders Detected</div>
            <div class="stat-value blue">{riders}</div>
        </div>""", unsafe_allow_html=True)

        stats_violations.markdown(f"""
        <div class="stat-card red">
            <div class="stat-label">Violations</div>
            <div class="stat-value red">{total_v}</div>
        </div>""", unsafe_allow_html=True)

        stats_plates.markdown(f"""
        <div class="stat-card yellow">
            <div class="stat-label">Plates Read</div>
            <div class="stat-value yellow">{total_p}</div>
        </div>""", unsafe_allow_html=True)

        stats_frames.markdown(f"""
        <div class="stat-card green">
            <div class="stat-label">Frames Processed</div>
            <div class="stat-value green">{frames}</div>
        </div>""", unsafe_allow_html=True)

    render_stats()

    st.markdown('<div class="section-header">🚨 Violations Log</div>', unsafe_allow_html=True)
    log_table = st.empty()

    st.markdown('<div class="section-header">🪪 Plates Detected</div>', unsafe_allow_html=True)
    plate_table = st.empty()

    def render_logs():
        if st.session_state.violations:
            df_v = pd.DataFrame(st.session_state.violations).tail(15)
            log_table.dataframe(
                df_v,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Time":       st.column_config.TextColumn("Time",   width="small"),
                    "Type":       st.column_config.TextColumn("Type",   width="medium"),
                    "Confidence": st.column_config.TextColumn("Conf.",  width="small"),
                    "Detail":     st.column_config.TextColumn("Detail", width="medium"),
                }
            )
        else:
            log_table.markdown(
                '<p style="color:#334155; font-size:0.8rem; text-align:center; padding:12px">No violations recorded</p>',
                unsafe_allow_html=True
            )

        if st.session_state.plates_detected:
            df_p = pd.DataFrame(st.session_state.plates_detected).tail(10)
            plate_table.dataframe(
                df_p,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Time":  st.column_config.TextColumn("Time",  width="small"),
                    "Plate": st.column_config.TextColumn("Plate", width="medium"),
                    "Conf.": st.column_config.TextColumn("Conf.", width="small"),
                }
            )
        else:
            plate_table.markdown(
                '<p style="color:#334155; font-size:0.8rem; text-align:center; padding:12px">No plates read yet</p>',
                unsafe_allow_html=True
            )

    render_logs()


# ---------------- PROCESS FRAME ----------------
def process_frame(frame, conf_threshold, show_conf):
    """Run full detection pipeline on a single frame."""
    detections = []

    # ── STEP 1: Detect riders & bikes using COCO model ──
    bike_results = bike_model(
        frame,
        classes=[PERSON_CLASS, MOTORCYCLE_CLASS],
        conf=conf_threshold,
        verbose=False
    )[0]

    riders_in_frame = 0

    for box in bike_results.boxes:
        cls_id = int(box.cls[0])
        conf   = float(box.conf[0])
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        label  = "rider" if cls_id == PERSON_CLASS else "bike"

        if label == "rider":
            riders_in_frame += 1

        detections.append({
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "type":        label,
            "confidence":  conf,
            "has_helmet":  True,
            "helmet_conf": None,
            "plate_text":  None,
            "plate_conf":  None,
            "show_conf":   show_conf,
        })

    st.session_state.total_riders = max(st.session_state.total_riders, riders_in_frame)

    # ── STEP 2: Detect helmet / no-helmet using your trained model ──
    helmet_results = helmet_ai(frame, conf=conf_threshold, verbose=False)[0]

    for box in helmet_results.boxes:
        cls_id     = int(box.cls[0])
        conf       = float(box.conf[0])
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        label_name = helmet_ai.names[cls_id]   # "With Helmet" / "Without Helmet"

        has_helmet = (
            "helmet" in label_name.lower() and
            "without" not in label_name.lower()
        )

        detections.append({
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "type":        "helmet_check",
            "confidence":  conf,
            "has_helmet":  has_helmet,
            "helmet_conf": conf,
            "plate_text":  None,
            "plate_conf":  None,
            "show_conf":   show_conf,
        })

        # Log violation if no helmet detected
        if not has_helmet:
            violation = {
                "Time":       pd.Timestamp.now().strftime("%H:%M:%S"),
                "Type":       "No Helmet",
                "Confidence": f"{conf * 100:.1f}%",
                "Detail":     f"Rider without helmet ({label_name})",
            }
            if not is_duplicate(violation, st.session_state.violations):
                st.session_state.violations.append(violation)
                save_log(violation, os.path.join(LOG_DIR, "violations.csv"))

    # ── STEP 3: Detect number plates (if best.pt has a plate class) ──
    for box in helmet_results.boxes:
        cls_id     = int(box.cls[0])
        label_name = helmet_ai.names[cls_id]

        if "plate" not in label_name.lower():
            continue

        conf = float(box.conf[0])
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        crop = frame[int(y1):int(y2), int(x1):int(x2)]

        if crop.size == 0:
            continue

        try:
            gray      = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            scaled    = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
            _, thresh = cv2.threshold(scaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            ocr_results = ocr_reader.readtext(thresh)
            if not ocr_results:
                ocr_results = ocr_reader.readtext(crop)

            if ocr_results:
                best     = max(ocr_results, key=lambda x: x[2])
                text     = best[1].strip().upper()
                conf_ocr = best[2]

                detections.append({
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                    "type":        "number_plate",
                    "confidence":  conf,
                    "has_helmet":  True,
                    "helmet_conf": None,
                    "plate_text":  text,
                    "plate_conf":  conf_ocr,
                    "show_conf":   show_conf,
                })

                plate_entry = {
                    "Time":  pd.Timestamp.now().strftime("%H:%M:%S"),
                    "Plate": text,
                    "Conf.": f"{conf_ocr * 100:.1f}%",
                }
                recent = st.session_state.plates_detected[-5:]
                if not any(p["Plate"] == text for p in recent):
                    st.session_state.plates_detected.append(plate_entry)

        except Exception as e:
            print("OCR error:", e)

    output_frame = draw_overlay(frame.copy(), detections)
    return output_frame, detections


# ---------------- MAIN PIPELINE ----------------
if uploaded_file:
    tfile = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=os.path.splitext(uploaded_file.name)[1]
    )
    tfile.write(uploaded_file.read())
    file_path = tfile.name

    is_image = uploaded_file.type.startswith("image")

    with col_feed:
        if is_image:
            status = st.info("⚙️ Processing image…", icon="🔍")
        else:
            status = st.info("⚙️ Processing video — frames streamed live…", icon="🎬")

    cap      = cv2.VideoCapture(file_path)
    frame_n  = 0
    stop_btn = col_feed.button("⏹ Stop Processing", key="stop")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if stop_btn:
            break

        frame_n += 1

        if not is_image and frame_n % process_every_n != 0:
            continue

        st.session_state.frames_processed += 1

        output_frame, _ = process_frame(frame, conf_threshold, show_conf_scores)

        # Resize for display if very large
        h, w = output_frame.shape[:2]
        if w > 1280:
            scale        = 1280 / w
            output_frame = cv2.resize(output_frame, (1280, int(h * scale)))

        st_frame.image(
            cv2.cvtColor(output_frame, cv2.COLOR_BGR2RGB),
            use_container_width=True,
            caption=f"Frame {frame_n}" if not is_image else uploaded_file.name,
        )

        render_stats()
        render_logs()

        if is_image:
            break

    cap.release()

    import time
    time.sleep(0.1)

    try:
        os.remove(file_path)
    except PermissionError:
        pass

    with col_feed:
        status.success("✅ Processing complete!", icon="🟢")