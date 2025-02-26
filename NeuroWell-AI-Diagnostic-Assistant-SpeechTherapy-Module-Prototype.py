from fastapi import FastAPI, UploadFile, File
import speech_recognition as sr
import librosa
import numpy as np
import requests
import json
from pydantic import BaseModel
from transformers import pipeline

app = FastAPI()

# Sentiment & Emotion Analysis Pipeline
emotion_analyzer = pipeline("text-classification", model="bert-base-uncased-emotion")

class SpeechAnalysisResult(BaseModel):
    transcript: str
    pronunciation_score: float
    fluency_score: float
    emotion: str

# Google Speech-to-Text API
GOOGLE_API_KEY = "key-to-be-entered"

@app.post("/analyze_speech", response_model=SpeechAnalysisResult)
async def analyze_speech(file: UploadFile = File(...)):
    recognizer = sr.Recognizer()
    audio_data, sr_rate = librosa.load(file.file, sr=16000)
    
    # Convert audio to text
    with sr.AudioFile(file.file) as source:
        audio = recognizer.record(source)
    transcript = recognizer.recognize_google(audio)
    
    # Pronunciation & Fluency Analysis
    words = transcript.split()
    pronunciation_score = np.random.uniform(70, 100)  # Placeholder for actual phoneme analysis
    fluency_score = 100 - (len([w for w in words if w in ['um', 'uh', 'hmm']]) / len(words)) * 100
    
    # Emotion Detection
    emotion_result = emotion_analyzer(transcript)
    detected_emotion = emotion_result[0]['label']
    
    return {
        "transcript": transcript,
        "pronunciation_score": pronunciation_score,
        "fluency_score": fluency_score,
        "emotion": detected_emotion
    }

@app.get("/generate_report/{user_id}")
def generate_report(user_id: str):
    # Fetch user data from DB
    report = {
        "user_id": user_id,
        "progress": "Fluency improved by 10% this month",
        "recommended_exercises": ["Repeat complex sentences", "Focus on articulation"],
        "emotion_trends": "More confidence detected in speech"
    }
    return report
