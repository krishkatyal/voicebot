import streamlit as st
import torch
import os
import tempfile
import asyncio
import edge_tts
from typing import List
from faster_whisper import WhisperModel
import groq
import av
import numpy as np
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
from queue import Queue
import threading

# Configure page
st.set_page_config(
    page_title="AI Voice Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Styling
st.markdown("""
    <style>
    /* Modern color scheme */
    :root {
        --primary-color: #4C61F0;
        --background-color: #FFFFFF;
        --text-color: #333333;
    }
    
    .stApp {
        background-color: var(--background-color);
        color: var(--text-color);
    }
    
    /* Chat messages */
    .user-message, .assistant-message {
        padding: 1rem;
        border-radius: 15px;
        margin: 0.5rem 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
    
    .user-message {
        background: #E3F2FD;
        border-left: 4px solid var(--primary-color);
    }
    
    .assistant-message {
        background: #F3E5F5;
        border-left: 4px solid #9C27B0;
    }
    
    /* Input area styling */
    .input-container {
        display: flex;
        align-items: center;
        gap: 10px;
        background: white;
        padding: 10px;
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
    
    /* Audio elements */
    .stAudio {
        margin-top: 1rem;
    }
    
    /* Streamlit elements */
    .stButton>button {
        background-color: var(--primary-color);
        color: white;
        border-radius: 20px;
        padding: 0.5rem 1rem;
        border: none;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        background-color: #3D4EBF;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'history' not in st.session_state:
    st.session_state.history = []
if 'audio_buffer' not in st.session_state:
    st.session_state.audio_buffer = Queue()

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
            
            # Display audio player in Streamlit
            st.audio(temp_path, format='audio/mp3')
            
            os.unlink(temp_path)
            
        except Exception as e:
            st.error(f"🔇 Audio generation error: {str(e)}")
            st.write(text)

def transcribe_audio(audio_data: np.ndarray) -> str:
    if audio_data.size == 0 or not whisper_model:
        return ""
        
    try:
        # Save audio data to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
            temp_path = temp_file.name
            audio_data = (audio_data * 32767).astype(np.int16)
            import soundfile as sf
            sf.write(temp_path, audio_data, samplerate=16000)

        segments, _ = whisper_model.transcribe(
            temp_path,
            beam_size=1,
            word_timestamps=False,
            language='en',
            vad_filter=True
        )
        os.unlink(temp_path)
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

def process_audio():
    while True:
        if not st.session_state.audio_buffer.empty():
            audio_data = st.session_state.audio_buffer.get()
            text_input = transcribe_audio(audio_data)
            if text_input:
                process_response(text_input)
        else:
            break

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

# Audio processing callback
def audio_frame_callback(frame):
    sound = frame.to_ndarray()
    sound = sound.mean(axis=1)
    st.session_state.audio_buffer.put(sound)

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

# Audio input
st.write("Click 'Start' to begin recording. Click 'Stop' when you're done speaking.")
webrtc_ctx = webrtc_streamer(
    key="speech-to-text",
    mode=WebRtcMode.SENDONLY,
    audio_receiver_size=1024,
    rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}),
    media_stream_constraints={"video": False, "audio": True},
)

if webrtc_ctx.audio_receiver:
    sound_chunk = webrtc_ctx.audio_receiver.get_frames()
    if sound_chunk:
        for audio_frame in sound_chunk:
            audio_frame_callback(audio_frame)
        threading.Thread(target=process_audio, daemon=True).start()

# Text input
text_input = st.text_input("Or type your message here:", key="text_input")

# Handle text input
if text_input:
    process_response(text_input)
    st.experimental_rerun()

# Clear chat button
if st.button("🗑️ Clear Chat", key="clear", help="Clear all messages"):
    st.session_state.history = []
    st.experimental_rerun()

