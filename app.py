import streamlit as st
import whisper
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch
import os

# Load Models
@st.cache_resource
def load_models():
    # Whisper for Speech-to-Text
    stt_model = whisper.load_model("tiny") 
    
    # Direct loading for Translation (NLLB)
    model_name = "facebook/nllb-200-distilled-600M"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    
    return stt_model, tokenizer, model

stt_model, tokenizer, model = load_models()

st.title("English Speech to Yoruba Text")

audio_file = st.file_uploader("Upload English Audio", type=["wav", "mp3", "m4a"])

if audio_file:
    with open("temp_audio", "wb") as f:
        f.write(audio_file.read())

    try:
        # 1. Speech to Text
        st.info("Transcribing...")
        stt_result = stt_model.transcribe("temp_audio")
        english_text = stt_result["text"]
        st.subheader("English:")
        st.write(english_text)

        # 2. Translation
        st.info("Translating...")
        inputs = tokenizer(english_text, return_tensors="pt")
        
        # Manually set the Yoruba target language
        translated_tokens = model.generate(
            **inputs, 
            forced_bos_token_id=tokenizer.lang_code_to_id["yor_Latn"], 
            max_length=100
        )
        
        yoruba_text = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
        
        st.subheader("Yoruba:")
        st.success(yoruba_text)

    except Exception as e:
        st.error(f"Error: {e}")
    finally:
        if os.path.exists("temp_audio"):
            os.remove("temp_audio")
