import cv2
import numpy as np
import streamlit as st
import mediapipe as mp

# --- Professional UI Architecture ---
st.set_page_config(page_title="Vigil-Eye Dashboard", layout="wide")

st.title("Vigil-Eye: Real-Time Drowsiness Detection System")
st.subheader("Developer: Shivansh")
st.markdown("---")

# Layout splitting for a clean dashboard view
col_controls, col_display = st.columns([1, 3])

with col_controls:
    st.markdown("### System Configuration")
    close_thresh = st.slider("Eye Closure Threshold (EAR)", 0.15, 0.35, 0.22, 0.01)
    frame_thresh = st.slider("Consecutive Alert Frames Trigger", 5, 30, 15, 1)
    
    st.markdown("### Device Management")
    run_system = st.checkbox("Initialize Live Stream Camera", value=False)
    
    st.markdown("### Operational Status")
    status_indicator = st.empty()

# --- Initialize MediaPipe Solutions Safely ---
# Explicit direct path utilized to fix the Python 3.14 Submodule Import Bug
import mediapipe.python.solutions.face_mesh as mp_face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Landmark coordinate indexing maps
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

def get_distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def calculate_ear(landmarks, eye_indices):
    p1 = landmarks[eye_indices[0]]
    p2 = landmarks[eye_indices[1]]
    p3 = landmarks[eye_indices[2]]
    p4 = landmarks[eye_indices[3]]
    p5 = landmarks[eye_indices[4]]
    p6 = landmarks[eye_indices[5]]
    
    v1 = get_distance((p2.x, p2.y), (p6.x, p6.y))
    v2 = get_distance((p3.x, p3.y), (p5.x, p5.y))
    h = get_distance((p1.x, p1.y), (p4.x, p4.y))
    
    return (v1 + v2) / (2.0 * h)

flag = 0

with col_display:
    st_frame = st.empty()

if run_system:
    capture = cv2.VideoCapture(0)
    
    while capture.isOpened():
        ret, frame = capture.read()
        if not ret:
            status_indicator.error("SYSTEM ERROR: Failed to interface with the device camera viewport.")
            break
            
        h, w, _ = frame.shape
        
        # Professional UI On-Screen Burn-In Watermark
        cv2.putText(frame, "Vigil-Eye System | Developer: Shivansh", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)
        
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            
            left_ear = calculate_ear(landmarks, LEFT_EYE)
            right_ear = calculate_ear(landmarks, RIGHT_EYE)
            avg_ear = (left_ear + right_ear) / 2.0
            
            if avg_ear < close_thresh:
                flag += 1
                if flag >= frame_thresh:
                    # Clean visual alert without chaotic emoji markers
                    cv2.putText(frame, "CRITICAL: DROWSINESS DETECTED", (50, 90), 
                                cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 255), 2)
                    status_indicator.error("SYSTEM STATUS: ALERT - Drowsiness patterns identified.")
            else:
                flag = 0
                status_indicator.success("SYSTEM STATUS: Active and Monitoring.")
        else:
            status_indicator.warning("SYSTEM STATUS: Searching for face coordinates...")
                
        st_frame.image(rgb_frame, channels="RGB", use_container_width=True)
        
    capture.release()
else:
    status_indicator.info("System Standby. Toggle the checkbox configuration panel to begin streaming.")
