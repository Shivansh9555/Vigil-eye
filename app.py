import cv2
import numpy as np
import streamlit as st
import mediapipe as mp
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration, WebRtcMode

# --- Technical UI Architecture Configuration ---
st.set_page_config(
    page_title="Vigil-Eye Control Console",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Human-Engineered Professional CSS Overview
st.markdown("""
    <style>
        /* Base typography overrides */
        .main-title { font-size: 2.0rem; font-weight: 700; color: #F8FAFC; margin-bottom: 0.1rem; letter-spacing: -0.025em; }
        .system-meta { font-size: 0.9rem; color: #94A3B8; margin-bottom: 1.8rem; }
        .developer-badge { font-family: monospace; font-size: 0.8rem; background-color: #0F172A; padding: 3px 8px; border-radius: 4px; color: #38BDF8; border: 1px solid #1E293B; }
        
        /* Metric component card structure */
        .metric-block { background-color: #0B0F19; padding: 16px; border-radius: 6px; border: 1px solid #1E293B; margin-bottom: 12px; }
        .metric-label { font-size: 0.75rem; font-weight: 600; color: #64748B; letter-spacing: 0.05em; text-transform: uppercase; }
        .metric-val { font-size: 1.6rem; font-weight: 700; color: #F1F5F9; margin-top: 4px; font-family: monospace; }
        
        /* Streamlit default overrides for clean industrial look */
        div[data-testid="stSidebar"] { background-color: #090D16; border-right: 1px solid #1E293B; }
        .stSlider { padding-bottom: 15px; }
    </style>
""", unsafe_allow_html=True)

# --- Corporate Interface Header Layout ---
st.markdown('<div class="main-title">Vigil-Eye Terminal</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="system-meta">Automated Core Real-Time Drowsiness Tracking Node | '
    '<span class="developer-badge">Lead Dev: Shivansh</span></div>', 
    unsafe_allow_html=True
)

# --- Configuration Sidebar Layout ---
with st.sidebar:
    st.markdown("### System Constants")
    st.markdown("Algorithmic thresholds for real-time validation parameters.")
    
    close_thresh = st.slider(
        "EAR Threshold Limit", 
        min_value=0.15, max_value=0.35, value=0.22, step=0.01,
        help="The numerical Eye Aspect Ratio bound below which a frame defaults to a closed status."
    )
    frame_thresh = st.slider(
        "Consecutive Frame Upper Bound", 
        min_value=5, max_value=30, value=15, step=1,
        help="Required persistent frames matching the criteria prior to system alert execution."
    )
    
    st.markdown("---")
    st.markdown("### Networking Infrastructure")
    st.caption("Active Mode: Local client-side WebRTC engine tunnel. Camera capture is processed locally via browser frames.")

# --- Algorithmic Foundation Submodules ---
import mediapipe.python.solutions.face_mesh as mp_face_mesh

LEFT_EYE_MAP = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_MAP = [33, 160, 158, 133, 153, 144]

def calculate_vector_magnitude(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def evaluate_eye_aspect_ratio(landmarks, indices):
    try:
        p1 = landmarks[indices[0]]
        p2 = landmarks[indices[1]]
        p3 = landmarks[indices[2]]
        p4 = landmarks[indices[3]]
        p5 = landmarks[indices[4]]
        p6 = landmarks[indices[5]]
        
        vertical_1 = calculate_vector_magnitude((p2.x, p2.y), (p6.x, p6.y))
        vertical_2 = calculate_vector_magnitude((p3.x, p3.y), (p5.x, p5.y))
        horizontal = calculate_vector_magnitude((p1.x, p1.y), (p4.x, p4.y))
        
        return (vertical_1 + vertical_2) / (2.0 * horizontal)
    except IndexError:
        return 0.28

# --- High-Performance Thread Safe Processing Subsystem ---
class ComputerVisionEngine(VideoProcessorBase):
    def __init__(self):
        self.mesh_processor = mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.consecutive_frames_counter = 0
        self.limit_ear = 0.22
        self.limit_frames = 15

    def update_runtime_variables(self, threshold_ear, threshold_frames):
        self.limit_ear = threshold_ear
        self.limit_frames = threshold_frames

    def recv(self, frame):
        image_matrix = frame.to_ndarray(format="bgr24")
        height, width, _ = image_matrix.shape
        
        # Clean architectural system stamp overlay
        cv2.putText(image_matrix, "SYSTEM CORE: VIGIL-EYE | DEV: SHIVANSH", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        
        processing_color_space = cv2.cvtColor(image_matrix, cv2.COLOR_BGR2RGB)
        mesh_results = self.mesh_processor.process(processing_color_space)
        
        if mesh_results.multi_face_landmarks:
            extracted_landmarks = mesh_results.multi_face_landmarks[0].landmark
            current_left_ear = evaluate_eye_aspect_ratio(extracted_landmarks, LEFT_EYE_MAP)
            current_right_ear = evaluate_eye_aspect_ratio(extracted_landmarks, RIGHT_EYE_MAP)
            computed_average_ear = (current_left_ear + current_right_ear) / 2.0
            
            # Non-intrusive runtime numerical diagnostics metrics placement
            cv2.putText(image_matrix, f"LIVE EAR BOUND: {computed_average_ear:.3f}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (148, 163, 184), 1, cv2.LINE_AA)
            
            if computed_average_ear < self.limit_ear:
                self.consecutive_frames_counter += 1
                if self.consecutive_frames_counter >= self.limit_frames:
                    # High contrast structured safety warning alert row layout 
                    cv2.rectangle(image_matrix, (15, height - 55), (width - 15, height - 15), (0, 0, 255), cv2.FILLED)
                    cv2.putText(image_matrix, "CRITICAL METRIC EXCEEDED: DROWSINESS SIGNATURE DETECTED", (30, height - 28), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            else:
                self.consecutive_frames_counter = 0
                cv2.putText(image_matrix, "STATE STATUS: NOMINAL (MONITORING ACTIVE)", (20, height - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (34, 197, 94), 1, cv2.LINE_AA)
        else:
            cv2.putText(image_matrix, "STATE STATUS: ERROR (SIGNAL ACQUISITION LOST)", (20, height - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (234, 179, 8), 1, cv2.LINE_AA)
            
        return frame.from_ndarray(image_matrix, format="bgr24")

# --- Interface Layout Partitioning ---
layout_viewport_panel, layout_telemetry_panel = st.columns([2.6, 1])

with layout_telemetry_panel:
    st.markdown("### Console Telemetry")
    
    st.markdown(
        f'<div class="metric-block">'
        f'<div class="metric-label">Assigned EAR Limit</div>'
        f'<div class="metric-val">{close_thresh:.2f}</div>'
        f'</div>', 
        unsafe_allow_html=True
    )
    
    st.markdown(
        f'<div class="metric-block">'
        f'<div class="metric-label">Alert Frame Limit</div>'
        f'<div class="metric-val">{frame_thresh} Frm</div>'
        f'</div>', 
        unsafe_allow_html=True
    )

with layout_viewport_panel:
    # Public STUN servers to guarantee network connection traversal across external devices
    network_stun_configuration = RTCConfiguration({"iceServers": [{"urls": ["stun:://google.com"]}]})
    
    stream_context = webrtc_streamer(
        key="vigil-eye-cv-node",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=network_stun_configuration,
        video_processor_factory=ComputerVisionEngine,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True
    )
    
    # Thread communication initialization
    if stream_context.video_processor:
        stream_context.video_processor.update_runtime_variables(close_thresh, frame_thresh)
