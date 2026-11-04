import av
import cv2
import numpy as np
import streamlit as st
import mediapipe as mp
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration, WebRtcMode

# --- Presentation UI Setup ---
st.set_page_config(page_title="Vigil-Eye Panel", layout="wide")

st.markdown("### Vigil-Eye: Drowsiness Detection System")
st.markdown("**Developer:** Shivansh | **System Status:** Ready for Evaluation")
st.markdown("---")

# Layout Partitioning for Presentation View
col_camera, col_telemetry = st.columns([2, 1])

with col_telemetry:
    st.markdown("#### Algorithmic Parameters")
    close_thresh = st.slider("EAR Threshold", 0.15, 0.35, 0.22, 0.01)
    frame_thresh = st.slider("Consecutive Frame Trigger", 5, 30, 10, 1)

    st.markdown("---")
    st.markdown("#### System Diagnostics")
    status_placeholder = st.empty()
    status_placeholder.info("Awaiting Camera Channel...")

# --- MediaPipe & Core Computer Vision Engine ---
mp_face_mesh = mp.solutions.face_mesh

LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]


class DrowsinessEngine(VideoProcessorBase):
    def __init__(self):
        self.face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=1, refine_landmarks=True,
            min_detection_confidence=0.5, min_tracking_confidence=0.5
        )
        self.counter = 0
        self.ear_limit = 0.22
        self.frame_limit = 10
        self.current_state = "STANDBY"

    def update_parameters(self, ear_limit, frame_limit):
        self.ear_limit = ear_limit
        self.frame_limit = frame_limit

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        h, w, _ = img.shape

        # Burn-in Watermark
        cv2.putText(img, "Vigil-Eye Engine | Dev: Shivansh", (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)

        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark

            # Helper lambda for clean mathematical distance parsing
            dist = lambda p1, p2: np.linalg.norm(np.array([p1.x, p1.y]) - np.array([p2.x, p2.y]))

            # Left Eye Aspect Ratio
            l_v1 = dist(landmarks[LEFT_EYE[1]], landmarks[LEFT_EYE[5]])
            l_v2 = dist(landmarks[LEFT_EYE[2]], landmarks[LEFT_EYE[4]])
            l_h = dist(landmarks[LEFT_EYE[0]], landmarks[LEFT_EYE[3]])
            left_ear = (l_v1 + l_v2) / (2.0 * l_h)

            # Right Eye Aspect Ratio
            r_v1 = dist(landmarks[RIGHT_EYE[1]], landmarks[RIGHT_EYE[5]])
            r_v2 = dist(landmarks[RIGHT_EYE[2]], landmarks[RIGHT_EYE[4]])
            r_h = dist(landmarks[RIGHT_EYE[0]], landmarks[RIGHT_EYE[3]])
            right_ear = (r_v1 + r_v2) / (2.0 * r_h)

            avg_ear = (left_ear + right_ear) / 2.0
            cv2.putText(img, f"Live EAR: {avg_ear:.3f}", (20, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

            # Drowsiness logic processing boundary
            if avg_ear < self.ear_limit:
                self.counter += 1
                if self.counter >= self.frame_limit:
                    cv2.rectangle(img, (15, h - 55), (w - 15, h - 15), (0, 0, 255), cv2.FILLED)
                    cv2.putText(img, "CRITICAL: DROWSINESS DETECTED", (30, h - 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                    self.current_state = "ALERT"
            else:
                self.counter = 0
                cv2.putText(img, "STATE: ACTIVE & MONITORING", (20, h - 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)
                self.current_state = "NOMINAL"
        else:
            cv2.putText(img, "STATE: LOSS OF SIGNAL", (20, h - 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 165, 255), 1, cv2.LINE_AA)
            self.current_state = "LOST"

        return av.VideoFrame.from_ndarray(img, format="bgr24")


with col_camera:
    rtc_config = RTCConfiguration({
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    })

    ctx = webrtc_streamer(
        key="vigil-eye-node",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=rtc_config,
        video_processor_factory=DrowsinessEngine,
        media_stream_constraints={"video": {"width": 640, "height": 480, "frameRate": 15}, "audio": False},
        async_processing=True,
    )

    if ctx.video_processor:
        ctx.video_processor.update_parameters(close_thresh, frame_thresh)

        state = ctx.video_processor.current_state
        if state == "ALERT":
            status_placeholder.error("CRITICAL STATUS: Drowsiness Identified.")
        elif state == "NOMINAL":
            status_placeholder.success("SYSTEM STATUS: Active Monitoring.")
        elif state == "LOST":
            status_placeholder.warning("SYSTEM STATUS: Searching For Face Coordinates.")