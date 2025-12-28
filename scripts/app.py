import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import tempfile
import os
from PIL import Image 


# Load your trained model
MODEL_DIR = "models"
model_path = os.path.join(MODEL_DIR, "emotion_model.h5")
model = load_model(model_path)

# Emotion labels
label_map = {0:'angry', 1:'disgust', 2:'fear', 3:'happy', 4:'neutral', 5:'sad', 6:'surprise'}

# Prediction function
def predict_emotion(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    if len(faces) == 0:
        return img, "No face detected"

    for (x,y,w,h) in faces:
        roi = gray[y:y+h, x:x+w]
        roi = cv2.resize(roi, (48,48))
        roi = roi.astype("float32") / 255.0
        roi = np.expand_dims(roi, axis=(0,-1))

        preds = model.predict(roi)
        label = label_map[np.argmax(preds)]

        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)
        cv2.putText(img, label, (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,0,0), 2)

    return img, label

# Streamlit UI
st.title("🎭 Emotion Detection App")

# Upload image option
uploaded_file = st.file_uploader("Upload an Image", type=["jpg","jpeg","png"])
if uploaded_file is not None:
    # Convert uploaded file to OpenCV image
    image = Image.open(uploaded_file).convert("RGB")
    img = np.array(image)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    result_img, label = predict_emotion(img)
    st.image(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB), caption=f"Prediction: {label}")

# Camera option
import cv2
import streamlit as st

if st.button("Start Camera"):
    stframe = st.empty()  # placeholder for video frame
    cap = cv2.VideoCapture(0)

    stop_button = st.button("Stop Camera")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.warning("Failed to grab frame.")
            break

        # Run emotion prediction
        result_img, label = predict_emotion(frame)
        rgb_frame = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)

        # Update single frame in UI
        stframe.image(rgb_frame, caption=f"Prediction: {label}", width=720)

        # Check for stop button
        if stop_button:
            break

        # Check for keyboard quit keys
        key = cv2.waitKey(1) & 0xFF
        if key in [ord('q'), ord('e'), ord('z')]:
            break

    cap.release()
    cv2.destroyAllWindows()
