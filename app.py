import streamlit as st
import sounddevice as sd
import numpy as np
import wave
import torch
import os
import tempfile
import threading
import time
import asyncio
import edge_tts
from typing import List
from faster_whisper import WhisperModel
import groq
import pygame

# Configure page
st.set_page_config(
    page_title="AI Voice Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Styling (same as original)
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
if 'recording' not in st.session_state:
    st.session_state.recording = False
if 'audio_playing' not in st.session_state:
    st.session_state.audio_playing = False
if 'processed_text' not in st.session_state:
    st.session_state.processed_text = None

def initialize_audio():
    try:
        pygame.mixer.init(frequency=24000, size=-16, channels=1, buffer=512)
    except pygame.error:
        try:
            pygame.mixer.init()
        except pygame.error:
            st.warning("⚠️ Audio output might not work properly on this system.")

# Initialize audio
initialize_audio()

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
        self._stop_event = threading.Event()
        self.VOICE = "en-US-JennyNeural"
        self.audio_enabled = True
        
        try:
            pygame.mixer.get_init()
        except:
            self.audio_enabled = False
            st.warning("⚠️ Audio playback is not available on this system.")
    
    def stop(self):
        if self.audio_enabled:
            self._stop_event.set()
            try:
                pygame.mixer.music.stop()
            except:
                pass
    
    async def _generate_speech(self, text: str, output_file: str):
        communicate = edge_tts.Communicate(text, self.VOICE)
        await communicate.save(output_file)
    
    def play(self, text: str):
        if not text:
            return
            
        if not self.audio_enabled:
            st.warning("⚠️ Audio playback is not available. Displaying text only.")
            st.write(text)
            return
            
        self._stop_event.clear()
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as temp_file:
                temp_path = temp_file.name
            
            asyncio.run(self._generate_speech(text, temp_path))
            
            try:
                pygame.mixer.music.load(temp_path)
                pygame.mixer.music.play()
                
                while pygame.mixer.music.get_busy():
                    if self._stop_event.is_set():
                        pygame.mixer.music.stop()
                        break
                    time.sleep(0.1)
            except Exception as e:
                st.error(f"Playback error: {str(e)}")
                st.write(text)
            
            os.unlink(temp_path)
            
        except Exception as e:
            st.error(f"🔇 Audio generation error: {str(e)}")
            st.write(text)
        finally:
            st.session_state.audio_playing = False

class OptimizedAudioRecorder:
    def __init__(self):
        self.stream = None
        self.frames = []
        self.is_recording = False

    def start_recording(self):
        self.frames = []
        self.is_recording = True
        self.stream = sd.InputStream(samplerate=16000, channels=1, callback=self.callback)
        self.stream.start()

    def stop_recording(self):
        if not self.stream:
            return None
            
        self.is_recording = False
        self.stream.stop()
        self.stream.close()
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as f:
            wf = wave.open(f.name, 'wb')
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit audio
            wf.setframerate(16000)
            wf.writeframes(b''.join(self.frames))
            wf.close()
            return f.name

    def callback(self, indata, frames, time, status):
        if self.is_recording:
            self.frames.append(indata.copy())

@st.cache_data(ttl=300)
def transcribe_audio(audio_file: str) -> str:
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

def process_response(text_input: str, is_voice: bool = False):
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
        
        st.session_state.audio_playing = True
        player_thread = threading.Thread(
            target=st.session_state.audio_player.play,
            args=(response,),
            daemon=True
        )
        player_thread.start()
    except Exception as e:
        st.error(f"Error processing response: {str(e)}")

# Initialize components
if 'audio_recorder' not in st.session_state:
    st.session_state.audio_recorder = OptimizedAudioRecorder()
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
col1, col2 = st.columns([6, 1])

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
            st.session_state.audio_recorder.start_recording()
        else:
            st.session_state.recording = False
            with st.spinner("Processing..."):
                audio_file = st.session_state.audio_recorder.stop_recording()
                if audio_file:
                    user_text = transcribe_audio(audio_file)
                    if user_text:
                        process_response(user_text, is_voice=True)
            st.rerun()

# Handle text input
if text_input and text_input != st.session_state.processed_text:
    st.session_state.processed_text = text_input
    process_response(text_input)
    st.rerun()

# Clear chat button
if st.button("🗑️ Clear Chat", key="clear", help="Clear all messages"):
    if st.session_state.audio_playing:
        st.session_state.audio_player.stop()
    st.session_state.history = []
    st.session_state.processed_text = None
    st.rerun()

