import streamlit as st
import requests
import uuid
import yaml
from pathlib import Path
import base64
from streamlit_mic_recorder import mic_recorder
from elevenlabs import ElevenLabs
from groq import Groq
import time
from datetime import datetime

# --- Page Configuration ---
st.set_page_config(page_title="F1 Race Engineer AI", page_icon="🏎️", layout="wide")

# --- Session State Initialization ---
context_defaults = {
    "context_track": "Silverstone", "context_driver": "VER", "context_grid": 10,
    "context_air_temp": 25, "context_track_temp": 35, "context_pit_loss": 22,
    "context_weather": "Dry", "context_deg": "Medium", "context_safety_car": 15,
    "context_overtaking": 6, "context_soft": 2, "context_medium": 3, "context_hard": 2
}
simulation_defaults = {
    "sim_stints": "", "sim_race_time": "N/A", "sim_best_lap": "N/A",
    "sim_avg_lap": "N/A", "sim_worst_lap": "N/A"
}
# --- SIMPLIFIED STATE: Removed initialization flags ---
defaults = {
    "thread_id": str(uuid.uuid4()),
    "groq_api_key": "",
    "elevenlabs_api_key": "",
    "elevenlabs_voice_id": "UdxnYaYl6VpkFjq6xH78", # Default Voice ID
    "last_processed_audio": None,
    "autoplay_audio_b64": None,
    "text_messages": [],
    "voice_messages": [],
    "current_mode": "Text",
    "backend_address": "https://lamaq-f1-strategy-api.hf.space"
}

for key, value in {**context_defaults, **simulation_defaults, **defaults}.items():
    if key not in st.session_state:
        st.session_state[key] = value

if st.session_state.thread_id is None:
    st.session_state.thread_id = str(uuid.uuid4())
if "last_processed_audio" not in st.session_state:
    st.session_state.last_processed_audio = None
if "last_audio_timestamp" not in st.session_state:
    st.session_state.last_audio_timestamp = None

if "text_messages" not in st.session_state:
    st.session_state.text_messages = []

# --- Helper Functions ---
@st.cache_data
def load_config():
    """Loads track and driver configurations."""
    config_path = Path(__file__).parent.parent / "track_config.yaml"
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        st.error("Fatal Error: `track_config.yaml` not found.")
        return None, [], []
    track_list = sorted(config.get('tracks', {}).keys())
    driver_list = ['VER', 'DOO', 'ANT', 'PIA', 'RUS', 'ALB', 'NOR', 'SAI', 'LEC',
                   'TSU', 'OCO', 'HAD', 'HAM', 'BOR', 'HUL', 'STR', 'BEA', 'ALO',
                   'GAS', 'LAW', 'COL']
    return config, track_list, sorted(driver_list)

TRACK_CONFIG, TRACK_LIST, DRIVER_LIST = load_config()

# --- Handle Simulation Data Transfer from other pages ---
if st.session_state.get("from_simulator", False) and "sim_results" in st.session_state:
    results = st.session_state.sim_results
    st.session_state.sim_stints = ", ".join([f"{s['compound'].capitalize()}-{s['laps']}" for s in results["stints"]])
    st.session_state.sim_race_time = results.get("race_time", "N/A")
    st.session_state.sim_best_lap = results.get("best_lap", "N/A")
    st.session_state.sim_avg_lap = results.get("avg_lap", "N/A")
    st.session_state.sim_worst_lap = results.get("worst_lap", "N/A")
    del st.session_state.sim_results
    st.session_state.from_simulator = False

# --- Reworked Chat Processing Logic ---
def process_chat(user_message: str):
    """
    Handles sending the user message and full context to the stateless backend,
    then updates the local session state with the new history from the backend.
    """
    is_voice_mode = st.session_state.current_mode == "Voice"
    current_messages = st.session_state.voice_messages if is_voice_mode else st.session_state.text_messages
    
    # Add user message to the local state for immediate UI update
    current_messages.append({"role": "user", "content": user_message})

# ============================ THE FIX IS HERE ============================
    # This block now includes ALL context variables required by the backend's
    # prompt template. This prevents the backend from throwing an error.
    context_updates = {
        "track_name": st.session_state.context_track,
        "driver_name": st.session_state.context_driver,
        "grid_position": st.session_state.context_grid,
        "air_temp": st.session_state.context_air_temp,
        "track_temp": st.session_state.context_track_temp,
        "pit_loss_time": st.session_state.context_pit_loss,
        "weather_conditions": st.session_state.context_weather,
        "deg_profile": st.session_state.context_deg,
        "safety_car_probability": st.session_state.context_safety_car,
        "overtaking_difficulty": st.session_state.context_overtaking,
        "tire_allocation": f"{st.session_state.context_soft}S/{st.session_state.context_medium}M/{st.session_state.context_hard}H"
    }
    simulation_updates = {
        "stints": [s.strip() for s in st.session_state.sim_stints.split(',') if s.strip()],
        "predicted_race_time": st.session_state.sim_race_time,
        "best_lap_time": st.session_state.sim_best_lap,
        "average_lap_time": st.session_state.sim_avg_lap,
        "worst_lap_time": st.session_state.sim_worst_lap
    }
    # ========================== END OF FIX ===========================


    # 2. Create the payload, including the chat history up to the user's last message
    api_payload = {
        "message": user_message,
        "context_updates": context_updates,
        "simulation_updates": simulation_updates,
        "chat_history": current_messages[:-1] # Send history *before* the latest user message
    }

    # 3. Set up headers for the API key
    headers = {
        "Authorization": f"Bearer {st.session_state.groq_api_key}",
        "Content-Type": "application/json"
    }

    with st.spinner("Mach is thinking..."):
        try:
            chat_url = f"{st.session_state.backend_address}/chat"
            response = requests.post(chat_url, json=api_payload, headers=headers)

            if response.status_code == 200:
                response_data = response.json()
                ai_response_text = response_data.get("response", "Sorry, I had an issue.")
                updated_history = response_data.get("updated_history", [])

                if is_voice_mode:
                    # Create a copy of the updated history to modify
                    modified_history = list(updated_history)
                    # Add audio data to the last message (AI response)
                    if st.session_state.elevenlabs_api_key and modified_history:
                        with st.spinner("Generating audio response..."):
                            try:
                                el_client = ElevenLabs(api_key=st.session_state.elevenlabs_api_key)
                                audio_stream = el_client.text_to_speech.convert(
                                    voice_id=st.session_state.elevenlabs_voice_id, 
                                    output_format="mp3_44100_128",
                                    text=ai_response_text, 
                                    model_id="eleven_multilingual_v2",
                                )
                                audio_data = b"".join(audio_stream)
                                audio_b64 = base64.b64encode(audio_data).decode("utf-8")
                                # Add audio to the last message
                                modified_history[-1]['audio_b64'] = audio_b64
                                # Set the audio to be played automatically
                                st.session_state.autoplay_audio_b64 = audio_b64
                            except Exception as e:
                                st.error(f"ElevenLabs audio generation failed: {e}")
                    # Update session state with modified history
                    st.session_state.voice_messages = modified_history
                else:
                    st.session_state.text_messages = updated_history

            else:
                error_info = response.json()
                st.error(f"Backend Error ({response.status_code}): {error_info.get('error', 'Unknown error')}")
                current_messages.pop() # Remove the user's message if the call failed

        except requests.RequestException as e:
            st.error(f"Connection Error: Could not reach the backend. Details: {e}")
            current_messages.pop()

    st.rerun()

# --- Autoplay Audio Logic ---
if st.session_state.autoplay_audio_b64:
    audio_html = f'<audio src="data:audio/mp3;base64,{st.session_state.autoplay_audio_b64}" controls autoplay style="display:none;"></audio>'
    st.markdown(audio_html, unsafe_allow_html=True)
    st.session_state.autoplay_audio_b64 = None # Clear the flag after playing

# --- UI Layout ---
st.title("💬 Mach - F1 Race Engineer")
st.markdown("Chat with your AI race engineer to discuss strategy options")

with st.sidebar:
    st.header("API Configuration")
    st.info("Your API keys are stored in the session and are required for the chatbot to function.")
    groq_api_key_input = st.text_input("Groq API Key", type="password", value=st.session_state.get("groq_api_key", ""))
    elevenlabs_api_key_input = st.text_input("ElevenLabs API Key", type="password", value=st.session_state.get("elevenlabs_api_key", ""))
    elevenlabs_voice_id_input = st.text_input("ElevenLabs Voice ID", value=st.session_state.elevenlabs_voice_id)
    if st.button("💾 Save API Keys", use_container_width=True):
        st.session_state.groq_api_key = groq_api_key_input
        st.session_state.elevenlabs_api_key = elevenlabs_api_key_input
        st.session_state.elevenlabs_voice_id = elevenlabs_voice_id_input
        st.success("API Keys have been saved for the session.")

col1, col2 = st.columns([1, 2])

with col1:
    st.header("Strategy Context")
    with st.container(border=True):
        st.subheader("Race Setup")
        c1, c2 = st.columns(2)
        track = c1.selectbox("Track", TRACK_LIST, index=TRACK_LIST.index(st.session_state.context_track))
        driver = c2.selectbox("Driver", DRIVER_LIST, index=DRIVER_LIST.index(st.session_state.context_driver))
        c1, c2, c3 = st.columns(3)
        grid = c1.number_input("Grid Position", 1, 20, value=st.session_state.context_grid)
        air_temp = c2.number_input("Air Temp (°C)", 10, 50, value=st.session_state.context_air_temp)
        track_temp = c3.number_input("Track Temp (°C)", 15, 60, value=st.session_state.context_track_temp)
        c1, c2 = st.columns(2)
        pit_loss = c1.number_input("Pit Loss (s)", 18, 30, value=st.session_state.context_pit_loss)
        weather = c2.selectbox("Weather", ["Dry", "Rain", "Mixed"], index=["Dry", "Rain", "Mixed"].index(st.session_state.context_weather))
        deg = c1.selectbox("Degradation Profile", ["Low", "Medium", "High"], index=["Low", "Medium", "High"].index(st.session_state.context_deg))
        safety_car = c2.slider("Safety Car Probability (%)", 0, 100, value=st.session_state.context_safety_car)
        overtaking = st.slider("Overtaking Difficulty (1-10)", 1, 10, value=st.session_state.context_overtaking)
        st.subheader("Tire Allocation")
        c1, c2, c3 = st.columns(3)
        soft = c1.number_input("Soft Sets", 0, 10, value=st.session_state.context_soft)
        medium = c2.number_input("Medium Sets", 0, 10, value=st.session_state.context_medium)
        hard = c3.number_input("Hard Sets", 0, 10, value=st.session_state.context_hard)
    if st.button("💾 Save Context", use_container_width=True):
        st.session_state.context_track = track
        st.session_state.context_driver = driver
        st.session_state.context_grid = grid
        st.session_state.context_air_temp = air_temp
        st.session_state.context_track_temp = track_temp
        st.session_state.context_pit_loss = pit_loss
        st.session_state.context_weather = weather
        st.session_state.context_deg = deg
        st.session_state.context_safety_car = safety_car
        st.session_state.context_overtaking = overtaking
        st.session_state.context_soft = soft
        st.session_state.context_medium = medium
        st.session_state.context_hard = hard
        st.session_state.context_saved = True
        st.success("Context saved! This will be used for future messages.")

    with st.container(border=True):
        st.subheader("Simulation Data")
        st.text(f"Stints: {st.session_state.sim_stints}")
        st.text(f"Best Lap: {st.session_state.sim_best_lap}s")
        st.text(f"Avg Lap: {st.session_state.sim_avg_lap}s")
        st.text(f"Worst Lap: {st.session_state.sim_worst_lap}s")
        st.text(f"Predicted Total: {st.session_state.sim_race_time}s")

with col2:
    st.header("Chat with Mach")
    
    current_mode = st.radio("Input Mode:", ("Text", "Voice"), horizontal=True, key="current_mode", label_visibility="collapsed")
    
    chat_container = st.container(height=500, border=True)
    messages = st.session_state.voice_messages if current_mode == "Voice" else st.session_state.text_messages
    
    with chat_container:
        for message in messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                # Display the audio player if audio data exists for an assistant message
                if message["role"] == "assistant" and "audio_b64" in message:
                    st.audio(base64.b64decode(message["audio_b64"]), format="audio/mp3")

    # --- Input Area ---
    chat_disabled = not st.session_state.groq_api_key
    if chat_disabled:
        st.warning("Please enter your Groq API Key in the sidebar to start chatting.")

    if current_mode == "Text":
        if prompt := st.chat_input("What's our strategy plan?", disabled=chat_disabled):
            process_chat(prompt)
    
    elif current_mode == "Voice":
        if not st.session_state.elevenlabs_api_key or not st.session_state.elevenlabs_voice_id:
            st.warning("Please enter your ElevenLabs API key and Voice ID to enable voice responses.")
        
        audio_data = None
        # Conditionally render the mic recorder only if the chat is not disabled.
        if not chat_disabled:
            audio_data = mic_recorder(
                start_prompt="🎤 Start Recording", 
                stop_prompt="⏹️ Stop Recording", 
                key='voice_recorder'
            )
        
        if audio_data and audio_data['bytes'] != st.session_state.last_processed_audio:
            st.session_state.last_processed_audio = audio_data['bytes']
            with st.spinner("Transcribing your message..."):
                try:
                    groq_client = Groq(api_key=st.session_state.groq_api_key)
                    transcription = groq_client.audio.transcriptions.create(
                        file=("mic_audio.wav", audio_data['bytes']),
                        model="whisper-large-v3"
                    ).text
                    if transcription:
                        process_chat(transcription)
                    else:
                        st.warning("Could not transcribe audio. Please try again.")
                except Exception as e:
                    st.error(f"Transcription failed: {e}")
