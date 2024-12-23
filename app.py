import streamlit as st
import torch
import os
import tempfile
import time
import asyncio
import edge_tts
from typing import List
from faster_whisper import WhisperModel
import groq
import base64
from datetime import datetime, timedelta

# Configure page
st.set_page_config(
    page_title="AI Voice Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Styling (unchanged)
st.markdown("""
    <style>
    /* Modern color scheme */
    :root {
        --primary-color: #4C61F0;
        --background-color: #FFFFFF;
    }
    
    .stApp {
        background-color: #F8F9FA;
    }
    
    /* Chat messages */
    .user-message {
        background: white;
        padding: 1rem;
        border-radius: 15px;
        margin: 0.5rem 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        border-left: 4px solid #4C61F0;
    }
    
    .assistant-message {
        background: #F8F9FA;
        padding: 1rem;
        border-radius: 15px;
        margin: 0.5rem 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        border-left: 4px solid #88C0D0;
    }
    
    /* Input area styling */
    .input-container {
        display: flex;
        align-items: center;
        gap: 10px;
        background: white;
        padding: 5px;
        border-radius: 15px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        margin-top: 1rem;
    }
    
    .stTextInput > div > div > input {
        border-radius: 12px;
        border: 2px solid #E9ECEF;
        padding: 0.5rem;
        font-size: 1rem;
    }
    
    /* Recording button */
    .record-button {
        background: none;
        border: none;
        color: #4C61F0;
        cursor: pointer;
        padding: 8px;
        border-radius: 50%;
        transition: all 0.3s ease;
    }
    
    .record-button:hover {
        background: rgba(76, 97, 240, 0.1);
    }
    
    .recording {
        color: #FF4B4B !important;
        animation: pulse 1.5s infinite;
    }
    
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.1); }
        100% { transform: scale(1); }
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'history' not in st.session_state:
    st.session_state.history = []
if 'processed_text' not in st.session_state:
    st.session_state.processed_text = None
if 'recording' not in st.session_state:
    st.session_state.recording = False
if 'record_start_time' not in st.session_state:
    st.session_state.record_start_time = None

# Initialize clients with provided keys
@st.cache_resource
def initialize_clients():
    try:
        os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
        
        groq_client = groq.Groq(
            api_key="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"
        )
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        whisper_model = WhisperModel(
            model_size_or_path="base.en",
            device=device,
            compute_type="float16" if device == 'cuda' else "float32"
        )
        
        return groq_client, whisper_model
    except Exception as e:
        st.error(f"Error initializing clients: {str(e)}")
        return None, None

groq_client, whisper_model = initialize_clients()

class OptimizedAudioPlayer:
    def __init__(self):
        self.VOICE = "en-US-JennyNeural"
    
    async def _generate_speech(self, text: str, output_file: str):
        communicate = edge_tts.Communicate(text, self.VOICE)
        await communicate.save(output_file)
    
    def play(self, text: str):
        if not text:
            return
        
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as temp_file:
                temp_path = temp_file.name
            
            asyncio.run(self._generate_speech(text, temp_path))
            
            # Read the audio file and encode it to base64
            with open(temp_path, "rb") as audio_file:
                audio_bytes = audio_file.read()
            audio_base64 = base64.b64encode(audio_bytes).decode()
            
            # Display audio player in Streamlit
            st.audio(audio_base64, format='audio/mp3')
            
            os.unlink(temp_path)
            
        except Exception as e:
            st.error(f"🔇 Audio generation error: {str(e)}")
            st.write(text)

@st.cache_data(ttl=300)
def transcribe_audio(audio_file) -> str:
    if not audio_file or not whisper_model:
        return ""
        
    try:
        segments, _ = whisper_model.transcribe(
            audio_file,
            beam_size=1,
            word_timestamps=False,
            language='en',
            vad_filter=True
        )
        return " ".join(segment.text for segment in segments)
    except Exception as e:
        st.error(f"Transcription error: {str(e)}")
        return ""

def get_assistant_response(messages: List[dict]) -> str:
    if not groq_client:
        return "Error: Groq client not properly initialized"
        
    try:
        completion = groq_client.chat.completions.create(
            model="mixtral-8x7b-32768",
            messages=messages,
            temperature=0.7,
            max_tokens=150,
        )
        return completion.choices[0].message.content
    except Exception as e:
        st.error(f"Groq API Error: {str(e)}")
        return f"Error: {str(e)}"

def process_response(text_input: str):
    if not text_input:
        return
        
    st.session_state.history.append({"role": "user", "content": text_input})
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Keep responses concise."}
    ] + st.session_state.history[-6:]
    
    try:
        response = get_assistant_response(messages)
        if response.startswith("Error:"):
            st.error(response)
            return
            
        st.session_state.history.append({"role": "assistant", "content": response})
        
        st.session_state.audio_player.play(response)
    except Exception as e:
        st.error(f"Error processing response: {str(e)}")

# Initialize components
if 'audio_player' not in st.session_state:
    st.session_state.audio_player = OptimizedAudioPlayer()

# Main UI
st.title("🤖 AI Voice Assistant")

# Chat container
chat_container = st.container()
with chat_container:
    for message in st.session_state.history:
        role_class = "user-message" if message["role"] == "user" else "assistant-message"
        st.markdown(
            f"""<div class="{role_class}">
                <strong>{'You' if message["role"] == "user" else '🤖 Assistant'}</strong>: {message["content"]}
                </div>""",
            unsafe_allow_html=True
        )

# Input area
col1, col2, col3 = st.columns([5, 1, 1])

with col1:
    text_input = st.text_input(
        "",
        placeholder="Type your message here...",
        key="text_input",
        label_visibility="collapsed"
    )

with col2:
    record_button_class = "record-button recording" if st.session_state.recording else "record-button"
    record_icon = "⏺️" if not st.session_state.recording else "⏹️"
    
    if st.button(record_icon, key="record_button"):
        if not st.session_state.recording:
            st.session_state.recording = True
            st.session_state.record_start_time = datetime.now()
        else:
            st.session_state.recording = False
            st.session_state.record_start_time = None
        st.rerun()

with col3:
    uploaded_file = st.file_uploader("Upload Audio", type=['wav', 'mp3'], key="audio_upload")

# Display recording timer
if st.session_state.recording:
    elapsed_time = datetime.now() - st.session_state.record_start_time
    st.write(f"Recording: {elapsed_time.seconds}.{elapsed_time.microseconds // 100000:01d}s")

# Handle text input
if text_input and text_input != st.session_state.processed_text:
    st.session_state.processed_text = text_input
    process_response(text_input)
    st.rerun()

# Handle audio file upload
if uploaded_file is not None:
    with st.spinner("Transcribing audio..."):
        # Save the uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
            temp_file.write(uploaded_file.getvalue())
            temp_file_path = temp_file.name

        # Transcribe the audio
        transcription = transcribe_audio(temp_file_path)

        # Remove the temporary file
        os.unlink(temp_file_path)

        if transcription:
            st.write("Transcription:", transcription)
            process_response(transcription)
            st.rerun()
        else:
            st.error("Failed to transcribe the audio. Please try again.")

# Clear chat button
if st.button("🗑️ Clear Chat", key="clear", help="Clear all messages"):
    st.session_state.history = []
    st.session_state.processed_text = None
    st.rerun()

