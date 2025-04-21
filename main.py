import streamlit as st
import numpy as np
import tensorflow as tf
import cv2
import os
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import LSTM, GlobalAveragePooling2D
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import Sequential
import firebase_admin
from firebase_admin import credentials, firestore

# Ensure TensorFlow is running in eager mode
tf.config.run_functions_eagerly(True)

# -------------------- SETUP --------------------
if not firebase_admin._apps:
    cred = credentials.Certificate("resources/serviceAccountKey.json")
    firebase_admin.initialize_app(cred)

db = firestore.client()

# -------------------- CONSTANTS --------------------
SEQUENCE_LENGTH = 30
IMAGE_SIZE = (224, 224)
MODEL_PATH = os.path.join("models", "seizure_lstm_model.h5")

# -------------------- DATABASE CLASS --------------------
class SeizureDatabase:
    def __init__(self):
        self.collection = db.collection('seizure_predictions')

    def add_prediction(self, video_name, label, confidence):
        try:
            self.collection.add({
                'video_name': str(os.path.basename(video_name)),
                'predicted_label': str(label),
                'confidence': float(confidence),
                'timestamp': firestore.SERVER_TIMESTAMP
            })
            return True
        except Exception as e:
            st.error(f"Failed to save prediction: {str(e)}")
            return False

    def get_predictions(self):
        try:
            docs = self.collection.order_by('timestamp', direction=firestore.Query.DESCENDING).stream()
            return pd.DataFrame([doc.to_dict() for doc in docs])
        except Exception as e:
            st.error(f"Failed to fetch predictions: {str(e)}")
            return pd.DataFrame()

# -------------------- MODEL LOADING --------------------
@st.cache_resource
def load_seizure_model():
    def custom_lstm_layer(**kwargs):
        kwargs.pop('time_major', None)
        return LSTM(**kwargs)

    try:
        model = load_model(
            MODEL_PATH,
            custom_objects={'LSTM': custom_lstm_layer}
        )
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        dummy_input = np.zeros((1, SEQUENCE_LENGTH, 1280), dtype=np.float32)
        model.predict(dummy_input, verbose=0)
        return model
    except Exception as e:
        st.error(f"Model loading failed: {str(e)}")
        return None

# -------------------- FEATURE EXTRACTOR --------------------
@st.cache_resource
def load_feature_extractor():
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3)
    )
    base_model.trainable = False
    model = Sequential([
        base_model,
        GlobalAveragePooling2D()
    ])
    dummy_img = np.zeros((1, 224, 224, 3), dtype=np.float32)
    model.predict(dummy_img, verbose=0)
    return model

# -------------------- VIDEO PROCESSING --------------------
def extract_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, IMAGE_SIZE)
        frame = tf.keras.applications.mobilenet_v2.preprocess_input(frame)
        frames.append(frame)
    cap.release()
    return frames

def create_sequences(frames):
    return [
        frames[i:i + SEQUENCE_LENGTH]
        for i in range(0, len(frames) - SEQUENCE_LENGTH + 1, SEQUENCE_LENGTH)
    ]

def extract_features(sequences, feature_extractor):
    if not sequences:
        return np.array([])
    
    sequences = np.array(sequences, dtype=np.float32)
    batch_size, seq_len, h, w, c = sequences.shape
    features = []
    
    for seq in sequences:
        flat = seq.reshape(-1, h, w, c)
        feat = feature_extractor.predict(flat, verbose=0)
        features.append(feat.reshape(seq_len, -1))
    
    return np.array(features, dtype=np.float32)

# -------------------- PREDICTION --------------------
def predict_seizure(video_path, model, feature_extractor, db):
    try:
        frames = extract_frames(video_path)
        if len(frames) < SEQUENCE_LENGTH:
            return "Error: Video too short (needs at least 30 frames)", None, None

        sequences = create_sequences(frames)
        if not sequences:
            return "Error: Could not create sequences", None, None

        features = extract_features(sequences, feature_extractor)
        if features.size == 0:
            return "Error: Feature extraction failed", None, None

        preds = model.predict(features, verbose=0)
        avg_pred = np.mean(preds, axis=0)
        class_idx = int(np.argmax(avg_pred))

        # Manual label map
        label_map = {0: 'No_Seizure', 1: 'P', 2: 'PG'}
        label = label_map.get(class_idx, str(class_idx))
        confidence = float(avg_pred[class_idx])

        db.add_prediction(video_path, label, confidence)
        return f"Prediction: {label} (Confidence: {confidence:.2%})", label, confidence

    except Exception as e:
        return f"Error during prediction: {str(e)}", None, None

# -------------------- STREAMLIT APP --------------------
def main():
    st.set_page_config(
        page_title="Seizure Detection System",
        layout="wide",
        page_icon="⚕️"
    )

    with st.spinner("Loading resources..."):
        model = load_seizure_model()
        feature_extractor = load_feature_extractor()
        seizure_db = SeizureDatabase()

    if model is None:
        st.error("Critical Error: Could not load required resources")
        st.stop()

    st.title("⚡ Seizure Detection from Video")
    st.markdown("Upload a video file to analyze for seizure activity.")

    uploaded_file = st.file_uploader(
        "Choose a video file", 
        type=["mp4", "avi", "mov"]
    )

    if uploaded_file:
        temp_dir = "temp"
        os.makedirs(temp_dir, exist_ok=True)
        video_path = os.path.join(temp_dir, uploaded_file.name)

        with open(video_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        col1, col2 = st.columns(2)
        with col1:
            st.video(video_path)

        with col2:
            with st.spinner("Analyzing video..."):
                result, label, confidence = predict_seizure(
                    video_path, model, feature_extractor, seizure_db)
            
            if label:
                st.success(result)
                st.metric("Prediction", label)
                st.metric("Confidence", f"{confidence:.2%}")
            else:
                st.error(result)

        try:
            os.remove(video_path)
        except:
            pass

    st.sidebar.header("Prediction History")
    if st.sidebar.button("Refresh History"):
        history = seizure_db.get_predictions()
        if not history.empty:
            st.sidebar.dataframe(history)
            st.sidebar.download_button(
                "Download CSV",
                history.to_csv(index=False),
                "seizure_history.csv"
            )
        else:
            st.sidebar.info("No history available")

if __name__ == "__main__":
    main()
