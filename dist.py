# THE ACTIONEERS - SAFE DIVE HACK 2025
# Driver Distraction Alert System - Streamlit Version

# -------------------------------------
# 1. IMPORT LIBRARIES
# -------------------------------------
import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import pyttsx3
import time
import streamlit as st  # Import Streamlit

print("Libraries imported successfully.")
print(f"OpenCV Version: {cv2.__version__}")
print(f"TensorFlow Version: {tf.__version__}")
print(f"Mediapipe Version: {mp.__version__}")

# -------------------------------------
# 2. INITIALIZE MODELS AND UTILITIES
# -------------------------------------

# --- Mediapipe Initialization ---
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5,
                                  min_tracking_confidence=0.5)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

# --- Text-to-Speech (TTS) Engine Initialization ---
try:
    tts_engine = pyttsx3.init()
    print("TTS Engine initialized.")
except Exception as e:
    print(f"Could not initialize TTS engine: {e}")
    tts_engine = None

# --- Load the Fine-Tuned TensorFlow/Keras Model ---
# IMPORTANT: Replace 'path/to/your/distraction_model.h5' with the actual path to your trained model file.
try:
    # We are using a placeholder model. You should replace this with your actual model.
    # model = tf.keras.models.load_model('path/to/your/distraction_model.h5')
    # For demonstration purposes, we will simulate the model's output.
    model = None
    print("Model placeholder created. Replace with: tf.keras.models.load_model('your_model.h5')")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Please ensure the model path is correct and the file is not corrupted.")
    model = None  # Set model to None if loading fails

# -------------------------------------
# 3. CONFIGURATION AND CONSTANTS
# -------------------------------------
# Labels should match the classes your model was trained on.
CLASS_LABELS = [
    'safe_driving',
    'texting',
    'talking_on_phone',
    'drinking',
    'reaching_behind',
    'yawning'
]

# Frame dimensions for model input
IMG_HEIGHT, IMG_WIDTH = 224, 224

# Alerting configuration
ALERT_COOLDOWN_SECONDS = 3
last_alert_time = 0
current_alert_message = ""


# -------------------------------------
# 4. HELPER FUNCTIONS
# -------------------------------------

def speak_alert(message):
    """Triggers a text-to-speech alert if the cooldown has passed."""
    global last_alert_time
    current_time = time.time()
    if (current_time - last_alert_time) > ALERT_COOLDOWN_SECONDS:
        if tts_engine:
            try:
                tts_engine.stop()
                tts_engine.say(message)
                tts_engine.runAndWait()
                last_alert_time = current_time
            except Exception as e:
                print(f"TTS Error: {e}")
        else:
            print(f"AUDIO ALERT: {message}")
        last_alert_time = current_time


def calculate_mouth_aspect_ratio(face_landmarks, frame_shape):
    """Calculates the Mouth Aspect Ratio (MAR) to detect yawning."""
    mouth_top_idx = 13
    mouth_bottom_idx = 14
    mouth_left_idx = 78
    mouth_right_idx = 308

    h, w = frame_shape
    mouth_top = (int(face_landmarks.landmark[mouth_top_idx].x * w), int(face_landmarks.landmark[mouth_top_idx].y * h))
    mouth_bottom = (int(face_landmarks.landmark[mouth_bottom_idx].x * w),
                    int(face_landmarks.landmark[mouth_bottom_idx].y * h))
    mouth_left = (int(face_landmarks.landmark[mouth_left_idx].x * w),
                  int(face_landmarks.landmark[mouth_left_idx].y * h))
    mouth_right = (int(face_landmarks.landmark[mouth_right_idx].x * w),
                   int(face_landmarks.landmark[mouth_right_idx].y * h))

    vertical_dist = np.linalg.norm(np.array(mouth_top) - np.array(mouth_bottom))
    horizontal_dist = np.linalg.norm(np.array(mouth_left) - np.array(mouth_right))

    if horizontal_dist == 0:
        return 0

    mar = vertical_dist / horizontal_dist
    return mar


def is_hand_near_face(hand_landmarks, face_landmarks, frame_shape):
    """Checks if a hand is close to the face region."""
    if not hand_landmarks or not face_landmarks:
        return False

    h, w = frame_shape

    face_x = [lm.x * w for lm in face_landmarks.landmark]
    face_y = [lm.y * h for lm in face_landmarks.landmark]
    face_min_x, face_max_x = min(face_x), max(face_x)
    face_min_y, face_max_y = min(face_y), max(face_y)

    for hand_lm in hand_landmarks:
        for lm in hand_lm.landmark:
            hand_x, hand_y = int(lm.x * w), int(lm.y * h)
            if (face_min_x < hand_x < face_max_x) and (face_min_y < hand_y < face_max_y):
                return True
    return False


# -------------------------------------
# 5. STREAMLIT APP LAYOUT
# -------------------------------------
st.set_page_config(page_title="Safe Dive - Driver Distraction Alert", layout="wide")
st.title("Safe Dive - Driver Distraction Alert System")
st.info("This application uses your webcam to monitor driver behavior and alert for distractions.")

run = st.checkbox('Run Webcam')
FRAME_WINDOW = st.image([])

# -------------------------------------
# 6. MAIN VIDEO PROCESSING LOOP
# -------------------------------------
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    st.error("Error: Could not open video source. Please check camera permissions.")
else:
    st.success("Video source opened successfully.")

while run:
    success, frame = cap.read()
    if not success:
        st.error("Failed to capture frame from webcam. Please restart.")
        break

    frame = cv2.flip(frame, 1)
    display_frame = frame.copy()
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # --- Model Prediction ---
    distraction_label = "safe_driving"
    if model:
        model_frame = cv2.resize(rgb_frame, (IMG_HEIGHT, IMG_WIDTH))
        model_frame = np.expand_dims(model_frame, axis=0)
        model_frame = model_frame / 255.0

        predictions = model.predict(model_frame)
        predicted_index = np.argmax(predictions[0])
        confidence = np.max(predictions[0])

        if confidence > 0.6:
            distraction_label = CLASS_LABELS[predicted_index]

    # --- Mediapipe Processing ---
    face_results = face_mesh.process(rgb_frame)
    hand_results = hands.process(rgb_frame)

    # --- Analysis and Alert Logic ---
    is_distracted = False
    alert_text = ""

    if face_results.multi_face_landmarks:
        for face_landmarks in face_results.multi_face_landmarks:
            mar = calculate_mouth_aspect_ratio(face_landmarks, frame.shape[:2])
            if mar > 0.6:
                alert_text = "ALERT: Yawning Detected"
                is_distracted = True
                speak_alert("Driver may be drowsy.")

    if not is_distracted and hand_results.multi_hand_landmarks and face_results.multi_face_landmarks:
        if is_hand_near_face(hand_results.multi_hand_landmarks, face_results.multi_face_landmarks[0], frame.shape[:2]):
            if distraction_label in ['texting', 'talking_on_phone']:
                alert_text = f"ALERT: {distraction_label.replace('_', ' ').title()}"
            else:
                alert_text = "ALERT: Hand near face"
            is_distracted = True
            speak_alert("Potential phone usage detected.")

    if not is_distracted and distraction_label != 'safe_driving':
        alert_text = f"ALERT: {distraction_label.replace('_', ' ').title()}"
        is_distracted = True
        speak_alert("Distraction detected.")

    # --- Display Information on Frame ---
    if is_distracted:
        status_text = f"STATUS: DISTRACTED ({alert_text})"
        status_color = (0, 0, 255)  # Red
    else:
        status_text = "STATUS: SAFE DRIVING"
        status_color = (0, 255, 0)  # Green

    cv2.rectangle(display_frame, (0, 0), (display_frame.shape[1], 40), (0, 0, 0), -1)
    cv2.putText(display_frame, status_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)

    if face_results.multi_face_landmarks:
        for face_landmarks in face_results.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                image=display_frame,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=drawing_spec,
                connection_drawing_spec=drawing_spec)
    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image=display_frame,
                landmark_list=hand_landmarks,
                connections=mp_hands.HAND_CONNECTIONS)

    # --- Show the final frame in Streamlit ---
    # Use BGR channel order as display_frame is from OpenCV
    FRAME_WINDOW.image(display_frame, channels="BGR")

# -------------------------------------
# 7. CLEANUP
# -------------------------------------
cap.release()
if tts_engine:
    tts_engine.stop()
st.write("Webcam stopped.")
print("Resources released. Program finished.")