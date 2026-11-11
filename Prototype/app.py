import streamlit as st
import librosa
import joblib
from pathlib import Path

MODEL_PATH = Path(__file__).parent / 'emotion_model.pkl'
model = joblib.load(MODEL_PATH)

emotion_names = {'ANG': 'Angry', 'DIS': 'Disgust', 'FEA': 'Fear',
                 'HAP': 'Happy', 'NEU': 'Neutral', 'SAD': 'Sad'}

st.title('Customer Emotion Detector')
st.write('Upload a short voice clip (.wav) and the model will predict the emotion.')

audio_file = st.file_uploader('Upload audio', type=['wav'])

if audio_file is not None:
    st.audio(audio_file)

    audio_file.seek(0)
    signal, sr = librosa.load(audio_file, sr=16000)
    mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=40)
    features = mfcc.mean(axis=1).reshape(1, -1)

    prediction = model.predict(features)[0]
    st.subheader(f'Predicted emotion: {emotion_names[prediction]}')
    