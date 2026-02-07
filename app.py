import streamlit as st
from st_audiorec import st_audiorec
import speech_recognition as sr
import io
from deep_translator import GoogleTranslator

st.title("🌍 English to Yoruba Translator")

# The recorder component
wav_audio_data = st_audiorec()

if wav_audio_data is not None:
    r = sr.Recognizer()
    # Convert bytes to an audio file object the library can read
    audio_file = io.BytesIO(wav_audio_data)
    
    with sr.AudioFile(audio_file) as source:
        audio = r.record(source)
        try:
            # English Transcription
            eng_text = r.recognize_google(audio)
            st.info(f"English: {eng_text}")

            # Yoruba Translation
            yor_text = GoogleTranslator(source='en', target='yo').translate(eng_text)
            st.success(f"Yoruba: {yor_text}")
        except Exception as e:
            st.error("Please speak more clearly or check your internet.")
