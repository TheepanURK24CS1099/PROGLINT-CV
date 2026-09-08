import os
import sys
import tempfile
import cv2
import streamlit as st
import numpy as np

from detector import PersonDetector
from tracker import PersonTracker
from metrics import TrackingMetrics
from pipeline import process_frame

# ==========================================
# PAGE CONFIGURATION & STYLING
# ==========================================
st.set_page_config(
    page_title="YOLOv8 Person Tracker",
    page_icon="👤",
    layout="centered"
)

# Custom minimal CSS for clean presentation
st.markdown("""
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        text-align: center;
        margin-bottom: 0.5rem;
        color: #1E293B;
    }
    .sub-header {
        font-size: 1rem;
        text-align: center;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 12px;
        text-align: center;
    }
    .metric-value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #0F172A;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #64748B;
    }
    .info-box {
        background-color: #F1F5F9;
        border-left: 4px solid #3B82F6;
        padding: 10px 14px;
        border-radius: 4px;
        font-size: 0.9rem;
        margin-bottom: 1rem;
    }
    </style>
""", unsafe_allow_html=True)


# ==========================================
# RESOURCE INITIALIZATION & CACHING
# ==========================================
@st.cache_resource
def load_detector(model_path: str = "models/best.pt"):
    """Loads and caches the YOLOv8 PersonDetector."""
    try:
        return PersonDetector(model_path)
    except FileNotFoundError as e:
        return None
    except Exception as e:
        return None


def main():
    st.markdown('<div class="main-header">YOLOv8 PERSON TRACKER</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Real-Time Person Detection & ByteTrack Multi-Object Tracking</div>', unsafe_allow_html=True)

    # 1. Load YOLOv8 Model
    model_path = "models/best.pt"
    if not os.path.exists(model_path) and os.path.exists("best.pt"):
        model_path = "best.pt"

    detector = load_detector(model_path)

    # Model not found error handling
    if detector is None:
        st.error("YOLOv8 model not found. Please check the model path.")
        st.info(f"Expected model path: `{os.path.abspath(model_path)}`")
        return

    model_info = detector.get_model_info()

    # Display Model Info summary
    st.markdown(f"""
        <div class="info-box">
            <b>Model:</b> Custom YOLOv8 (<code>{os.path.basename(model_info['model_path'])}</code>) &nbsp;|&nbsp; 
            <b>Tracker:</b> ByteTrack &nbsp;|&nbsp; 
            <b>Class:</b> Person &nbsp;|&nbsp; 
            <b>Device:</b> {model_info['device']}
        </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # ==========================================
    # INPUT SOURCE SELECTION
    # ==========================================
    source_type = st.radio(
        "Select Input Source:",
        options=["Upload Video", "Start Webcam"],
        horizontal=True,
        key="source_type_radio"
    )

    video_path = None
    if source_type == "Upload Video":
        uploaded_file = st.file_uploader(
            "Upload a video file (.mp4, .avi, .mov, .mkv)",
            type=["mp4", "avi", "mov", "mkv"],
            key="video_uploader"
        )
        if uploaded_file is not None:
            # Save uploaded file temporarily to input directory
            os.makedirs("input", exist_ok=True)
            temp_video_path = os.path.join("input", "uploaded_temp.mp4")
            with open(temp_video_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            video_path = temp_video_path

    # ==========================================
    # CONFIDENCE THRESHOLD & CONTROL BUTTONS
    # ==========================================
    st.markdown("---")
    conf_threshold = st.slider(
        "Confidence Threshold:",
        min_value=0.10,
        max_value=0.95,
        value=0.50,
        step=0.05,
        key="confidence_slider"
    )

    col_start, col_stop = st.columns(2)
    with col_start:
        start_button = st.button("Start Tracking", use_container_width=True, type="primary", key="start_btn")
    with col_stop:
        stop_button = st.button("Stop Tracking", use_container_width=True, key="stop_btn")

    # Session state for tracking control
    if "tracking_active" not in st.session_state:
        st.session_state.tracking_active = False

    if start_button:
        st.session_state.tracking_active = True
    if stop_button:
        st.session_state.tracking_active = False

    # Video display placeholder & metrics placeholders
    st.markdown("---")
    video_placeholder = st.empty()
    
    col_m1, col_m2, col_m3 = st.columns(3)
    with col_m1:
        metric1_ph = st.empty()
    with col_m2:
        metric2_ph = st.empty()
    with col_m3:
        metric3_ph = st.empty()

    # Helper function to display metrics
    def update_metrics_ui(curr, total, fps_val):
        metric1_ph.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{curr}</div>
                <div class="metric-label">People Currently Tracked</div>
            </div>
        """, unsafe_allow_html=True)
        metric2_ph.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{total}</div>
                <div class="metric-label">Total Unique People</div>
            </div>
        """, unsafe_allow_html=True)
        metric3_ph.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{fps_val:.1f}</div>
                <div class="metric-label">Processing FPS</div>
            </div>
        """, unsafe_allow_html=True)

    # Initial metrics display
    update_metrics_ui(0, 0, 0.0)

    # ==========================================
    # TRACKING PROCESSING LOOP
    # ==========================================
    if st.session_state.tracking_active:
        tracker = PersonTracker(track_thresh=conf_threshold)
        metrics = TrackingMetrics()

        if source_type == "Upload Video":
            if not video_path or not os.path.exists(video_path):
                st.error("Unable to open video. Please select a valid video file.")
                st.session_state.tracking_active = False
                return

            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                st.error("Unable to open video. Please select a valid video file.")
                st.session_state.tracking_active = False
                return
        else:
            # Webcam input
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                st.error("Unable to access webcam. Please check camera permissions.")
                st.session_state.tracking_active = False
                return

        # Main processing frame-by-frame loop
        while cap.isOpened() and st.session_state.tracking_active:
            ret, frame = cap.read()
            if not ret:
                break

            # Process frame using unified pipeline
            annotated_frame, frame_metrics = process_frame(
                frame, detector, tracker, metrics, conf_threshold=conf_threshold
            )

            # Convert BGR (OpenCV) to RGB (Streamlit)
            frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            video_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

            # Update Metrics UI
            update_metrics_ui(
                frame_metrics['current_people'],
                frame_metrics['total_unique'],
                frame_metrics['fps']
            )

        cap.release()
        st.session_state.tracking_active = False
        st.success("Tracking completed or stopped.")


# ==========================================
# DIRECT CLI / OPENCV WINDOW FALLBACK MODE
# ==========================================
def run_cli_opencv():
    """Runs direct OpenCV GUI window when launched directly via python command."""
    print("Launching YOLOv8 Person Tracker in OpenCV GUI mode...")
    model_path = "models/best.pt" if os.path.exists("models/best.pt") else "best.pt"
    
    try:
        detector = PersonDetector(model_path)
    except Exception as e:
        print(f"YOLOv8 model not found. Please check the model path: {e}")
        return

    tracker = PersonTracker()
    metrics = TrackingMetrics()

    # Check for video path in input folder or webcam
    input_files = [f for f in os.listdir("input") if f.endswith(('.mp4', '.avi', '.mov', '.mkv'))] if os.path.exists("input") else []
    
    if input_files:
        src = os.path.join("input", input_files[0])
        print(f"Opening video file: {src}")
        cap = cv2.VideoCapture(src)
    else:
        print("Opening Webcam (Device 0)...")
        cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Unable to open video or webcam source.")
        return

    print("Press 'q' to quit.")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        annotated_frame, m = process_frame(frame, detector, tracker, metrics, conf_threshold=0.5)

        # Overlay metrics on frame in CLI mode
        cv2.putText(annotated_frame, f"People Tracked: {m['current_people']} | Total Unique: {m['total_unique']} | FPS: {m['fps']:.1f}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("YOLOv8 Person Tracker", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        if get_script_run_ctx() is not None:
            main()
        else:
            run_cli_opencv()
    except Exception as e:
        main()
