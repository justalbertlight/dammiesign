import streamlit as st
from st_audiorec import st_audiorec  # New component for browser recording
import speech_recognition as sr
import io
from deep_translator import GoogleTranslator

st.set_page_config(page_title="FYP: English-Yoruba", page_icon="🌍")

st.title("🌍 English to Yoruba Speech Translator")
st.write("Record your voice below to translate to Yoruba.")

# 1. Browser-based Recording
wav_audio_data = st_audiorec()

if wav_audio_data is not None:
    # 2. Process the audio in memory
    r = sr.Recognizer()
    audio_file = io.BytesIO(wav_audio_data)
    
    with sr.AudioFile(audio_file) as source:
        audio = r.record(source)
        
        try:
            with st.spinner("Transcribing..."):
                # 3. Speech to Text
                english_text = r.recognize_google(audio)
                st.subheader("English Transcription:")
                st.info(english_text)

                # 4. Translation
                translated = GoogleTranslator(source='en', target='yo').translate(english_text)
                st.subheader("Yoruba Translation:")
                st.success(translated)
                
        except Exception as e:
            st.error("Could not process audio. Please speak clearly.")
