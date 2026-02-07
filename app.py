import streamlit as st
import whisper
from transformers import pipeline
import os

# Load Models
@st.cache_resource
def load_models():
    # 'tiny' ensures the app doesn't crash from low memory
    stt_model = whisper.load_model("tiny") 
    
    # Using 'text2text-generation' avoids the KeyError: 'translation'
    translator = pipeline(
        "text2text-generation", 
        model="facebook/nllb-200-distilled-600M"
    )
    return stt_model, translator

stt_model, translator = load_models()

# UI Layout
st.title("English Speech to Yoruba Text")
st.markdown("Upload an English audio file to get the Yoruba translation.")

audio_file = st.file_uploader("Upload Audio", type=["wav", "mp3", "m4a"])

if audio_file:
    # Save temp file
    with open("temp_audio", "wb") as f:
        f.write(audio_file.read())

    try:
        # 1. Speech to Text
        st.info("Step 1: Transcribing English...")
        result = stt_model.transcribe("temp_audio")
        english_text = result["text"]
        
        st.subheader("English Transcription:")
        st.info(english_text)

        # 2. Translation
        st.info("Step 2: Translating to Yoruba...")
        # Note: 'forced_bos_token_id' or 'tgt_lang' is handled here
        translated = translator(
            english_text, 
            forced_bos_token_id=translator.tokenizer.lang_code_to_id["yor_Latn"],
            max_length=512
        )
        yoruba_text = translated[0]['generated_text']
        
        st.subheader("Yoruba Translation:")
        st.success(yoruba_text)

    except Exception as e:
        st.error(f"An error occurred: {e}")
    
    finally:
        if os.path.exists("temp_audio"):
            os.remove("temp_audio")
