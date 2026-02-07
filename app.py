import streamlit as st
import whisper
from transformers import pipeline
import os

# Load Models
@st.cache_resource
def load_models():
    stt_model = whisper.load_model("base")
    translator = pipeline("translation", model="facebook/nllb-200-distilled-600M", 
                          src_lang="eng_Latn", tgt_lang="yor_Latn")
    return stt_model, translator

stt_model, translator = load_models()

# UI Layout
st.title("English Speech to Yoruba Text")
audio_file = st.file_uploader("Upload English Audio", type=["wav", "mp3", "m4a"])

if audio_file:
    # Save temp file
    with open("temp_audio", "wb") as f:
        f.write(audio_file.read())

    # 1. Speech to Text
    st.info("Transcribing English...")
    result = stt_model.transcribe("temp_audio")
    english_text = result["text"]
    st.subheader("English Transcription:")
    st.write(english_text)

    # 2. Translation
    st.info("Translating to Yoruba...")
    translated = translator(english_text)
    yoruba_text = translated[0]['translation_text']
    
    st.subheader("Yoruba Translation:")
    st.success(yoruba_text)
    
    os.remove("temp_audio")

