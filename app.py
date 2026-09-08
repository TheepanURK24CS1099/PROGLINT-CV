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
# PAGE CONFIGURATION & DARK CV THEME
# ==========================================
st.set_page_config(
    page_title="YOLOv8 + ByteTrack Person Tracker",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional Computer-Vision Dashboard CSS
st.markdown("""
    <style>
    /* Dark Palette Core */
    .stApp {
        background-color: #0B0F17;
        color: #F8FAFC;
        font-family: 'Inter', system-ui, -apple-system, sans-serif;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #0E1420 !important;
        border-right: 1px solid #1F2937 !important;
    }
    section[data-testid="stSidebar"] .block-container {
        padding-top: 1.5rem !important;
        padding-left: 1rem !important;
        padding-right: 1rem !important;
    }

    /* Hide default padding & stream element chrome */
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        max-width: 1400px;
    }

    /* Header styling */
    .app-header {
        background: linear-gradient(90deg, #111827 0%, #1E293B 100%);
        border: 1px solid #1F2937;
        border-radius: 12px;
        padding: 16px 24px;
        margin-bottom: 20px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
    }
    .app-title {
        font-size: 1.6rem;
        font-weight: 800;
        letter-spacing: -0.025em;
        color: #F8FAFC;
        margin: 0;
        line-height: 1.2;
    }
    .app-subtitle {
        font-size: 0.85rem;
        font-weight: 500;
        color: #3B82F6;
        margin-top: 2px;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .status-badge {
        display: inline-flex;
        align-items: center;
        background-color: rgba(16, 185, 129, 0.15);
        color: #34D399;
        border: 1px solid rgba(16, 185, 129, 0.3);
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 600;
        margin-left: 12px;
    }
    .status-dot {
        height: 7px;
        width: 7px;
        background-color: #10B981;
        border-radius: 50%;
        display: inline-block;
        margin-right: 6px;
        box-shadow: 0 0 8px #10B981;
    }
    .header-info-pill {
        background-color: #1E293B;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 6px 14px;
        font-size: 0.82rem;
        color: #94A3B8;
    }
    .header-info-pill span {
        color: #F8FAFC;
        font-weight: 600;
    }

    /* Cards & Panels */
    .dashboard-card {
        background-color: #111827;
        border: 1px solid #1F2937;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.25);
    }
    .card-title {
        font-size: 0.95rem;
        font-weight: 700;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 16px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }

    /* Empty Video Display Box */
    .video-empty-state {
        background-color: #090D16;
        border: 2px dashed #1F2937;
        border-radius: 12px;
        height: 440px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        text-align: center;
        color: #64748B;
        padding: 20px;
    }
    .video-empty-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #94A3B8;
        margin-top: 12px;
        margin-bottom: 6px;
    }
    .video-empty-desc {
        font-size: 0.85rem;
        color: #475569;
        max-width: 320px;
    }

    /* Video Footer bar */
    .video-footer-bar {
        background-color: #090D16;
        border: 1px solid #1F2937;
        border-radius: 8px;
        padding: 8px 16px;
        margin-top: 10px;
        display: flex;
        justify-content: space-between;
        font-size: 0.82rem;
        color: #94A3B8;
        font-family: monospace;
    }

    /* Live Stat Cards */
    .stat-card {
        background-color: #111827;
        border: 1px solid #1F2937;
        border-radius: 12px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
    }
    .stat-value {
        font-size: 2.2rem;
        font-weight: 800;
        color: #3B82F6;
        line-height: 1.1;
        letter-spacing: -0.03em;
    }
    .stat-label {
        font-size: 0.78rem;
        font-weight: 600;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-top: 6px;
    }

    /* Key Value System Table */
    .sys-info-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 0.85rem;
    }
    .sys-info-table td {
        padding: 7px 0;
        border-bottom: 1px solid #1F2937;
    }
    .sys-info-table td.label {
        color: #64748B;
        font-weight: 500;
    }
    .sys-info-table td.val {
        color: #F8FAFC;
        font-weight: 600;
        text-align: right;
    }

    /* Tracked Persons Table */
    .track-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 0.85rem;
        text-align: left;
    }
    .track-table th {
        background-color: #1E293B;
        color: #94A3B8;
        font-weight: 600;
        padding: 10px 14px;
        text-transform: uppercase;
        font-size: 0.75rem;
        letter-spacing: 0.05em;
    }
    .track-table td {
        padding: 10px 14px;
        border-bottom: 1px solid #1F2937;
        color: #E2E8F0;
    }
    .track-table tr:hover {
        background-color: #1E293B;
    }
    .badge-tracking {
        background-color: rgba(16, 185, 129, 0.2);
        color: #34D399;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.75rem;
        font-weight: 600;
    }

    /* Streamlit Widget Styling Overrides */
    div[data-testid="stRadio"] > label {
        color: #94A3B8 !important;
        font-size: 0.82rem !important;
        font-weight: 600 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.05em !important;
    }
    div[data-testid="stSlider"] > label {
        color: #94A3B8 !important;
        font-size: 0.82rem !important;
        font-weight: 600 !important;
        text-transform: uppercase !important;
    }
    .stButton > button {
        border-radius: 8px !important;
        font-weight: 600 !important;
        transition: all 0.2s ease !important;
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
    except FileNotFoundError:
        return None
    except Exception:
        return None


def main():
    # 1. Load YOLOv8 Model
    model_path = "models/best.pt"
    if not os.path.exists(model_path) and os.path.exists("best.pt"):
        model_path = "best.pt"

    detector = load_detector(model_path)
    is_model_ready = detector is not None

    # ==========================================
    # HEADER SECTION
    # ==========================================
    header_html = f"""
    <div class="app-header">
        <div>
            <div style="display: flex; align-items: center;">
                <h1 class="app-title">PERSON TRACKER</h1>
                <div class="status-badge">
                    <span class="status-dot"></span>
                    {'System Ready' if is_model_ready else 'Model Missing'}
                </div>
            </div>
            <div class="app-subtitle">YOLOv8 Detection + ByteTrack Multi-Object Tracking</div>
        </div>
        <div class="header-info-pill">
            Model: <span>Custom YOLOv8</span> &nbsp;|&nbsp; Tracker: <span>ByteTrack</span>
        </div>
    </div>
    """
    st.markdown(header_html, unsafe_allow_html=True)

    # Model missing error banner if model is not loaded
    if not is_model_ready:
        st.error("Unable to load detection model. Please check the model path at 'models/best.pt' or 'best.pt'.")
        return

    model_info = detector.get_model_info()

    # Session state for tracking control & AI video playback
    if "tracking_active" not in st.session_state:
        st.session_state.tracking_active = False
    if "ai_video_path" not in st.session_state:
        st.session_state.ai_video_path = None

    # ==========================================
    # LEFT SLIDING SIDEBAR: CONTROL PANEL
    # ==========================================
    with st.sidebar:
        st.markdown('<div class="dashboard-card"><div class="card-title">CONTROL PANEL</div>', unsafe_allow_html=True)

        # 1. INPUT SOURCE SELECTION
        st.markdown('<div style="font-size: 0.8rem; font-weight: 700; color: #94A3B8; text-transform: uppercase; margin-bottom: 8px;">INPUT SOURCE</div>', unsafe_allow_html=True)
        source_type = st.radio(
            "Select Input Source",
            options=["Upload Video", "Webcam"],
            horizontal=True,
            key="input_source_radio",
            label_visibility="collapsed"
        )

        video_path = None
        video_metadata_str = ""

        if source_type == "Upload Video":
            uploaded_file = st.file_uploader(
                "Upload a video file (.mp4, .avi, .mov, .mkv)",
                type=["mp4", "avi", "mov", "mkv"],
                key="video_file_uploader_dark"
            )
            if uploaded_file is not None:
                os.makedirs("input", exist_ok=True)
                temp_video_path = os.path.join("input", "uploaded_temp.mp4")
                with open(temp_video_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                video_path = temp_video_path
                video_metadata_str = f"File: {uploaded_file.name} ({uploaded_file.size / (1024*1024):.1f} MB)"
                st.caption(f"📁 `{video_metadata_str}`")
        else:
            st.markdown("""
                <div style="background-color: rgba(16, 185, 129, 0.1); border: 1px solid rgba(16, 185, 129, 0.2); border-radius: 6px; padding: 8px 12px; font-size: 0.82rem; color: #34D399; margin-bottom: 12px;">
                    ● Camera Connected (Ready)
                </div>
            """, unsafe_allow_html=True)

        st.markdown("<hr style='border-color: #1F2937; margin: 16px 0;'>", unsafe_allow_html=True)

        # 2. DETECTION SETTINGS
        st.markdown('<div style="font-size: 0.8rem; font-weight: 700; color: #94A3B8; text-transform: uppercase; margin-bottom: 4px;">DETECTION SETTINGS</div>', unsafe_allow_html=True)
        conf_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.10,
            max_value=0.95,
            value=0.50,
            step=0.05,
            key="conf_slider_dark"
        )
        st.markdown(f'<div style="font-size: 0.8rem; color: #64748B; margin-top: -8px; margin-bottom: 14px;">Current threshold: <strong>{int(conf_threshold*100)}%</strong></div>', unsafe_allow_html=True)

        st.markdown("<hr style='border-color: #1F2937; margin: 16px 0;'>", unsafe_allow_html=True)

        # 3. TRACKING ACTION BUTTONS
        st.markdown('<div style="font-size: 0.8rem; font-weight: 700; color: #94A3B8; text-transform: uppercase; margin-bottom: 8px;">TRACKING CONTROL</div>', unsafe_allow_html=True)
        btn_col1, btn_col2 = st.columns(2)
        with btn_col1:
            start_btn = st.button("Start Tracking", use_container_width=True, type="primary", key="btn_start_dark")
        with btn_col2:
            stop_btn = st.button("Stop Tracking", use_container_width=True, key="btn_stop_dark")

        if start_btn:
            st.session_state.tracking_active = True
        if stop_btn:
            st.session_state.tracking_active = False

        st.markdown("</div>", unsafe_allow_html=True)

        # 4. SYSTEM INFORMATION CARD
        st.markdown(f"""
            <div class="dashboard-card">
                <div class="card-title">SYSTEM INFORMATION</div>
                <table class="sys-info-table">
                    <tr><td class="label">Model</td><td class="val">Custom YOLOv8</td></tr>
                    <tr><td class="label">Tracker</td><td class="val">ByteTrack</td></tr>
                    <tr><td class="label">Detection Class</td><td class="val">Person</td></tr>
                    <tr><td class="label">Hardware Device</td><td class="val">{model_info['device']}</td></tr>
                    <tr><td class="label">Confidence</td><td class="val">{int(conf_threshold*100)}%</td></tr>
                    <tr><td class="label">Input Source</td><td class="val">{'Webcam' if source_type == 'Webcam' else 'Video'}</td></tr>
                </table>
            </div>
        """, unsafe_allow_html=True)

    # ==========================================
    # MAIN AREA: VIDEO COMPARISON SECTION
    # ==========================================
    st.markdown("""
        <div style="text-align: center; margin: 0 0 16px 0; background: linear-gradient(90deg, #111827 0%, #1E293B 100%); border: 1px solid #1F2937; border-radius: 10px; padding: 10px;">
            <span style="font-size: 1.15rem; font-weight: 800; letter-spacing: 0.12em; color: #F8FAFC; text-transform: uppercase;">
                VIDEO COMPARISON
            </span>
        </div>
    """, unsafe_allow_html=True)

    col_orig, col_ai = st.columns(2)

    with col_orig:
        st.markdown("""
            <div class="dashboard-card" style="padding-bottom: 12px; margin-bottom: 10px;">
                <div class="card-title">
                    <span>ORIGINAL VIDEO</span>
                    <span style="color: #94A3B8; font-size: 0.78rem;">RAW INPUT</span>
                </div>
        """, unsafe_allow_html=True)
        orig_video_ph = st.empty()
        orig_footer_ph = st.empty()
        orig_footer_ph.markdown("""
            <div class="video-footer-bar">
                <span>Stream: <strong>Original</strong></span>
                <span>Type: <strong>Raw Video</strong></span>
            </div>
            </div>
        """, unsafe_allow_html=True)

    with col_ai:
        live_status_html = '<span style="color: #10B981; font-weight: 700;">● LIVE</span>' if st.session_state.tracking_active else '<span style="color: #64748B;">● STANDBY</span>'
        st.markdown(f"""
            <div class="dashboard-card" style="padding-bottom: 12px; margin-bottom: 10px;">
                <div class="card-title">
                    <span>AI-GENERATED VIDEO</span>
                    <span>{live_status_html}</span>
                </div>
        """, unsafe_allow_html=True)
        ai_video_ph = st.empty()
        ai_footer_ph = st.empty()
        ai_footer_ph.markdown("""
            <div class="video-footer-bar">
                <span>FPS: <strong>0.0</strong></span>
                <span>Resolution: <strong>-- × --</strong></span>
            </div>
            </div>
        """, unsafe_allow_html=True)

    # Initial / Standby display state
    if not st.session_state.tracking_active:
        if source_type == "Upload Video" and video_path and os.path.exists(video_path):
            orig_video_ph.video(video_path)
            if st.session_state.ai_video_path and os.path.exists(st.session_state.ai_video_path):
                ai_video_ph.video(st.session_state.ai_video_path)
            else:
                ai_video_ph.markdown("""
                    <div class="video-empty-state">
                        <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="#475569" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
                            <polygon points="23 7 16 12 23 17 23 7"></polygon>
                            <rect x="1" y="5" width="15" height="14" rx="2" ry="2"></rect>
                        </svg>
                        <div class="video-empty-title">Ready for AI Tracking</div>
                        <div class="video-empty-desc">Click <strong>Start Tracking</strong> in the control panel to generate side-by-side detections.</div>
                    </div>
                """, unsafe_allow_html=True)
        else:
            orig_video_ph.markdown("""
                <div class="video-empty-state">
                    <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="#475569" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
                        <polygon points="23 7 16 12 23 17 23 7"></polygon>
                        <rect x="1" y="5" width="15" height="14" rx="2" ry="2"></rect>
                    </svg>
                    <div class="video-empty-title">No video selected</div>
                    <div class="video-empty-desc">Upload a video file or select webcam from the left control panel.</div>
                </div>
            """, unsafe_allow_html=True)
            ai_video_ph.markdown("""
                <div class="video-empty-state">
                    <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="#475569" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
                        <polygon points="23 7 16 12 23 17 23 7"></polygon>
                        <rect x="1" y="5" width="15" height="14" rx="2" ry="2"></rect>
                    </svg>
                    <div class="video-empty-title">AI Video Output</div>
                    <div class="video-empty-desc">Detections with persistent IDs will stream here side-by-side.</div>
                </div>
            """, unsafe_allow_html=True)

    # ==========================================
    # LIVE STATISTICS CARDS (3 PRIMARY METRICS)
    # ==========================================
    stat_col1, stat_col2, stat_col3 = st.columns(3)
    with stat_col1:
        stat1_ph = st.empty()
    with stat_col2:
        stat2_ph = st.empty()
    with stat_col3:
        stat3_ph = st.empty()

    def update_stats_display(curr: int, total: int, fps_val: float):
        stat1_ph.markdown(f"""
            <div class="stat-card">
                <div class="stat-value">{curr:02d}</div>
                <div class="stat-label">CURRENT PEOPLE</div>
            </div>
        """, unsafe_allow_html=True)
        stat2_ph.markdown(f"""
            <div class="stat-card">
                <div class="stat-value">{total:02d}</div>
                <div class="stat-label">TOTAL UNIQUE</div>
            </div>
        """, unsafe_allow_html=True)
        stat3_ph.markdown(f"""
            <div class="stat-card">
                <div class="stat-value">{fps_val:.1f}</div>
                <div class="stat-label">FPS</div>
            </div>
        """, unsafe_allow_html=True)

    # Render initial stat cards
    update_stats_display(0, 0, 0.0)

    # ==========================================
    # TRACKED PERSONS TABLE SECTION
    # ==========================================
    st.markdown("""
        <div class="dashboard-card" style="margin-top: 20px;">
            <div class="card-title">TRACKED PERSONS</div>
    """, unsafe_allow_html=True)

    tracks_table_ph = st.empty()

    def update_tracks_table(tracks_list):
        if not tracks_list:
            tracks_table_ph.markdown("""
                <div style="text-align: center; color: #475569; padding: 20px; font-size: 0.85rem;">
                    No active tracks detected
                </div>
            """, unsafe_allow_html=True)
            return

        rows_html = ""
        for t in tracks_list:
            t_id = str(t.get('person_id', t['track_id']))
            t_conf = f"{t['conf']:.2f}"
            rows_html += f"""
                <tr>
                    <td><strong>{t_id}</strong></td>
                    <td>{t_conf}</td>
                    <td><span class="badge-tracking">Tracking</span></td>
                </tr>
            """

        tracks_table_ph.markdown(f"""
            <table class="track-table">
                <thead>
                    <tr>
                        <th>ID</th>
                        <th>CONFIDENCE</th>
                        <th>STATUS</th>
                    </tr>
                </thead>
                <tbody>
                    {rows_html}
                </tbody>
            </table>
        """, unsafe_allow_html=True)

    update_tracks_table([])
    st.markdown("</div>", unsafe_allow_html=True)

    # ==========================================
    # TRACKING FRAME PROCESSING LOOP
    # ==========================================
    if st.session_state.tracking_active:
        tracker = PersonTracker(track_thresh=conf_threshold)
        metrics = TrackingMetrics()

        if source_type == "Upload Video":
            if not video_path or not os.path.exists(video_path):
                st.error("No video file selected. Please upload a valid video file.")
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
                st.error("Camera unavailable. Please check camera permissions.")
                st.session_state.tracking_active = False
                return

        os.makedirs("output", exist_ok=True)
        raw_out_path = os.path.join("output", "temp_ai_tracked.mp4")
        final_ai_video = os.path.join("output", "ai_generated_playback.mp4")
        writer = None

        # Frame loop
        while cap.isOpened() and st.session_state.tracking_active:
            ret, frame = cap.read()
            if not ret:
                break

            h, w = frame.shape[:2]

            # Initialize video writer for AI output playback if processing a video file
            if writer is None and source_type == "Upload Video":
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                fps_val = cap.get(cv2.CAP_PROP_FPS) or 25.0
                writer = cv2.VideoWriter(raw_out_path, fourcc, fps_val, (w, h))

            # 1. Update Original Video Frame (Left Side)
            frame_orig_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            orig_video_ph.image(frame_orig_rgb, channels="RGB", use_container_width=True)

            # 2. Process frame using existing CV pipeline (Detection + Tracking + Re-ID)
            annotated_frame, frame_metrics = process_frame(
                frame, detector, tracker, metrics, conf_threshold=conf_threshold
            )

            if writer is not None:
                writer.write(annotated_frame)

            # 3. Update AI Annotated Frame (Right Side)
            annotated_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            ai_video_ph.image(annotated_rgb, channels="RGB", use_container_width=True)

            # 4. Update Footers
            orig_footer_ph.markdown(f"""
                <div class="video-footer-bar">
                    <span>Stream: <strong>Raw Input</strong></span>
                    <span>Resolution: <strong>{w} × {h}</strong></span>
                </div>
                </div>
            """, unsafe_allow_html=True)

            ai_footer_ph.markdown(f"""
                <div class="video-footer-bar">
                    <span>FPS: <strong>{frame_metrics['fps']:.1f}</strong></span>
                    <span>Active: <strong>{frame_metrics['current_people']}</strong></span>
                </div>
                </div>
            """, unsafe_allow_html=True)

            # 5. Update Statistics Cards & Track Table
            update_stats_display(
                frame_metrics['current_people'],
                frame_metrics['total_unique'],
                frame_metrics['fps']
            )
            update_tracks_table(frame_metrics.get('tracks', []))

        cap.release()
        if writer is not None:
            writer.release()
            # Fast convert with ffmpeg for seamless browser playback
            ffmpeg_path = "/Users/theepan/.local/bin/ffmpeg" if os.path.exists("/Users/theepan/.local/bin/ffmpeg") else "ffmpeg"
            os.system(f"{ffmpeg_path} -y -i {raw_out_path} -vcodec libx264 -pix_fmt yuv420p -movflags +faststart {final_ai_video} -loglevel quiet")
            if os.path.exists(final_ai_video):
                st.session_state.ai_video_path = final_ai_video
            else:
                st.session_state.ai_video_path = raw_out_path

        st.session_state.tracking_active = False
        st.rerun()


if __name__ == "__main__":
    main()

