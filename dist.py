# =========================
# Driver Distraction Detector (Streamlit)
# =========================
# Detects: Eyes-closed (drowsy), Yawning, Looking-away, Phone-use (+ bottle/cup)
# Works on macOS/Windows/Linux. No external sound files needed.
# Dependencies:
#   pip install streamlit opencv-python mediapipe ultralytics
# Place yolov8n.pt in the same directory (or give a full path below).

import os
import platform
import time
import tempfile

import cv2
import numpy as np
import streamlit as st
import mediapipe as mp
from ultralytics import YOLO

# -----------------------------
# 0) Streamlit CONFIG (MUST be first Streamlit call)
# -----------------------------
st.set_page_config(page_title="Driver Distraction Detection", layout="wide")
st.title("Advanced Driver Assistance System")

# -----------------------------
# 1) Cross-platform Beep (cooldown)
# -----------------------------
def play_beep():
    sys = platform.system()
    try:
        if sys == "Windows":
            import winsound
            winsound.Beep(1200, 250)  # freq, ms
        elif sys == "Darwin":  # macOS
            # Play built-in system sound instead of speaking "beep"
            os.system('afplay /System/Library/Sounds/Glass.aiff')
        else:  # Linux
            # Requires 'sox' for 'play' command. As a fallback, use console bell.
            code = os.system('play -nq -t alsa synth 0.2 sine 1000 2>/dev/null')
            if code != 0:
                print("\a", end="")
    except Exception:
        print("\a", end="")

# -----------------------------
# 2) UI Controls
# -----------------------------
with st.sidebar:
    st.header("Settings")
    source = st.radio("Video source", ["Webcam", "Upload video"])
    yolo_path = st.text_input("YOLOv8 model file", "yolov8n.pt")

    st.subheader("Thresholds")
    EAR_THRESH = st.slider("Eyes Closed EAR threshold", 0.10, 0.35, 0.21, 0.005)
    EAR_MIN_FRAMES = st.slider("Min frames: Eyes closed", 3, 60, 15, 1)

    MAR_THRESH = st.slider("Yawning MAR threshold", 0.30, 1.00, 0.60, 0.01)
    MAR_MIN_FRAMES = st.slider("Min frames: Yawn", 1, 45, 8, 1)

    HEAD_YAW_FRAC = st.slider(
        "Looking-away sensitivity (nose offset as % of face width)",
        0.05, 0.60, 0.25, 0.01
    )
    HEAD_MIN_FRAMES = st.slider("Min frames: Looking away", 2, 60, 8, 1)

    st.subheader("YOLO Detection")
    yolo_conf = st.slider("YOLO confidence", 0.10, 0.90, 0.50, 0.05)
    detect_phone = st.checkbox("Detect phone", True)
    detect_drink = st.checkbox("Detect drinking (bottle/cup)", True)

    st.subheader("Alert")
    enable_sound = st.checkbox("Enable beep", True)
    BEEP_COOLDOWN = st.slider("Beep cooldown (seconds)", 0.2, 5.0, 1.5, 0.1)

    start = st.checkbox("Start detection")

# -----------------------------
# 3) Prepare Video Source
# -----------------------------
video_path = None
if source == "Upload video":
    uploaded_file = st.file_uploader("Upload video", type=["mp4", "avi", "mov", "mkv"])
    if uploaded_file:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        video_path = tfile.name
else:
    video_path = 0  # webcam

# -----------------------------
# 4) Load YOLO (optional but recommended)
# -----------------------------
yolo_model = None
yolo_loaded_ok = False
yolo_warnings = []
if os.path.exists(yolo_path):
    try:
        yolo_model = YOLO(yolo_path)
        yolo_loaded_ok = True
    except Exception as e:
        yolo_warnings.append(f"YOLO load failed: {e}")
else:
    yolo_warnings.append(f"Model file not found: {yolo_path}")

if yolo_warnings:
    st.warning(" | ".join(yolo_warnings))

# -----------------------------
# 5) Mediapipe Setup
# -----------------------------
mp_face_mesh = mp.solutions.face_mesh

# Indices for EAR/MAR using MediaPipe FaceMesh (468 landmarks)
LEFT_EYE = [33, 160, 158, 133, 153, 144]   # [p1, p2, p3, p6, p5, p4]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH_VERT_TOP = 13
MOUTH_VERT_BOTTOM = 14
MOUTH_HORZ_LEFT = 78
MOUTH_HORZ_RIGHT = 308

def ear_from_landmarks(lms, ids):
    p1 = np.array([lms[ids[0]].x, lms[ids[0]].y])
    p2 = np.array([lms[ids[1]].x, lms[ids[1]].y])
    p3 = np.array([lms[ids[2]].x, lms[ids[2]].y])
    p4 = np.array([lms[ids[5]].x, lms[ids[5]].y])
    p5 = np.array([lms[ids[4]].x, lms[ids[4]].y])
    p6 = np.array([lms[ids[3]].x, lms[ids[3]].y])
    v = np.linalg.norm(p2 - p4) + np.linalg.norm(p3 - p5)
    h = np.linalg.norm(p1 - p6) + 1e-6
    return v / (2.0 * h)

def mar_from_landmarks(lms):
    top = np.array([lms[MOUTH_VERT_TOP].x, lms[MOUTH_VERT_TOP].y])
    bot = np.array([lms[MOUTH_VERT_BOTTOM].x, lms[MOUTH_VERT_BOTTOM].y])
    left = np.array([lms[MOUTH_HORZ_LEFT].x, lms[MOUTH_HORZ_LEFT].y])
    right = np.array([lms[MOUTH_HORZ_RIGHT].x, lms[MOUTH_HORZ_RIGHT].y])
    v = np.linalg.norm(top - bot)
    h = np.linalg.norm(left - right) + 1e-6
    return v / h

def head_yaw_fraction(lms, img_w):
    nose_idx = 1
    left_eye_idx = 33
    right_eye_idx = 263
    nose_x = lms[nose_idx].x * img_w
    left_x = lms[left_eye_idx].x * img_w
    right_x = lms[right_eye_idx].x * img_w
    mid_x = 0.5 * (left_x + right_x)
    face_width = abs(right_x - left_x) + 1e-6
    return abs(nose_x - mid_x) / face_width

# -----------------------------
# 6) Display holders
# -----------------------------
frame_area = st.empty()
status_area = st.empty()
hint_area = st.empty()

# -----------------------------
# 7) Detection loop
# -----------------------------
if start and video_path is not None:
    cap = cv2.VideoCapture(video_path)

    eyes_closed_frames = 0
    yawn_frames = 0
    away_frames = 0
    last_beep_time = 0.0

    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as face_mesh:

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_results = face_mesh.process(rgb)
            active_alerts = []

            if face_results.multi_face_landmarks:
                lms = face_results.multi_face_landmarks[0].landmark
                xs = np.array([lm.x for lm in lms]); ys = np.array([lm.y for lm in lms])
                x1 = int(xs.min() * w); y1 = int(ys.min() * h)
                x2 = int(xs.max() * w); y2 = int(ys.max() * h)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (80, 200, 80), 2)

                left_ear = ear_from_landmarks(lms, LEFT_EYE)
                right_ear = ear_from_landmarks(lms, RIGHT_EYE)
                ear = 0.5 * (left_ear + right_ear)
                if ear < EAR_THRESH: eyes_closed_frames += 1
                else: eyes_closed_frames = 0
                if eyes_closed_frames >= EAR_MIN_FRAMES:
                    active_alerts.append("😴 Eyes Closed (Drowsy)")

                mar = mar_from_landmarks(lms)
                if mar > MAR_THRESH: yawn_frames += 1
                else: yawn_frames = 0
                if yawn_frames >= MAR_MIN_FRAMES:
                    active_alerts.append("😮 Yawning")

                yaw_frac = head_yaw_fraction(lms, w)
                if yaw_frac > HEAD_YAW_FRAC: away_frames += 1
                else: away_frames = 0
                if away_frames >= HEAD_MIN_FRAMES:
                    active_alerts.append("↔️ Looking Away")

                cv2.putText(frame, f"EAR:{ear:.2f}  MAR:{mar:.2f}  Yaw:{yaw_frac:.2f}",
                            (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
            else:
                away_frames += 1
                if away_frames >= HEAD_MIN_FRAMES:
                    active_alerts.append("🟡 Face Not Visible")

            if yolo_loaded_ok and (detect_phone or detect_drink):
                try:
                    yout = yolo_model.predict(frame, conf=yolo_conf, imgsz=416, verbose=False)
                    for r in yout:
                        for box in r.boxes:
                            cls = int(box.cls[0])
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            if detect_phone and cls == 67:
                                active_alerts.append("📱 Phone Detected")
                                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                                cv2.putText(frame, "Phone", (x1, y1 - 6),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                            if detect_drink and cls in (39, 41):
                                active_alerts.append("🥤 Drinking Detected")
                                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 160, 0), 2)
                                cv2.putText(frame, "Drink", (x1, y1 - 6),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 160, 0), 2)
                except Exception as e:
                    cv2.putText(frame, f"YOLO error: {e}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

            if active_alerts:
                msg = " | ".join(sorted(set(active_alerts)))
                status_area.error(f"🚨 {msg}")
                now = time.time()
                if enable_sound and (now - last_beep_time) >= BEEP_COOLDOWN:
                    play_beep()
                    last_beep_time = now
            else:
                status_area.success("✅ Focused on driving")

            frame_area.image(frame, channels="BGR")

            if not st.session_state.get("keep_running", True) and source == "Webcam":
                break

    cap.release()
else:
    hint_area.info("✅ Set options in the sidebar and tick **Start detection** to begin.")
