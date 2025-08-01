import streamlit as st
import requests
import pandas as pd
import re
import yaml
from pathlib import Path

# --- Page Configuration ---
st.set_page_config(page_title="F1 Strategy Simulator", page_icon="📈", layout="wide")

# --- Global Configuration ---
BACKEND_ADDRESS = "https://lamaq-f1-strategy-api.hf.space"
if "backend_address" not in st.session_state:
    st.session_state.backend_address = BACKEND_ADDRESS

# --- Helper Functions ---
@st.cache_data
def load_sim_config():
    """Loads track and driver configuration from a YAML file."""
    config_path = Path(__file__).parent / "track_config.yaml"
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        st.error("Fatal Error: `track_config.yaml` not found.")
        st.stop()
        return None, [], []
    track_data = {track: data.get('total_laps', 55) for track, data in config.get('tracks', {}).items()}
    driver_list = ['VER', 'DOO', 'ANT', 'PIA', 'RUS', 'ALB', 'NOR', 'SAI', 'LEC',
                   'TSU', 'OCO', 'HAD', 'HAM', 'BOR', 'HUL', 'STR', 'BEA', 'ALO',
                   'GAS', 'LAW', 'COL']
    return track_data, sorted(track_data.keys()), sorted(driver_list)

# --- Load Initial Data ---
TRACK_DATA, TRACK_LIST, DRIVER_LIST = load_sim_config()

# --- State Management for Simulator ---
# This dictionary defines the initial state for all widgets on this page.
sim_defaults = {
    "sim_track": "Silverstone",
    "sim_driver": "VER",
    "sim_num_stops": 1,
    "sim_custom_weather": False,
    "sim_air_temp": 25.0,
    "sim_track_temp": 35.0,
    "simulation_results_exist": False, # Use a simple flag
    "from_simulator": False,
}
# Add defaults for the maximum possible number of stints.
for i in range(4):
    sim_defaults[f"c_{i}"] = "MEDIUM"
    sim_defaults[f"l_{i}"] = 20

# This loop ensures that the state is initialized only once.
for key, value in sim_defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

# --- Page Title ---
st.title("📈 F1 Race Strategy Simulator")
st.markdown("Design a pit stop strategy and simulate a race to see the predicted lap times.")

# --- Main UI for Simulation ---
col1, col2 = st.columns(2)
with col1:
    st.subheader("Race Setup")
    # Each widget is now explicitly bound to its key in the session state.
    st.selectbox("Select Track:", TRACK_LIST, key="sim_track")
    st.selectbox("Select Driver:", DRIVER_LIST, key="sim_driver")
    total_laps = TRACK_DATA.get(st.session_state.sim_track, 55)
    st.metric("Official Race Laps", total_laps)
    if st.checkbox("Set Custom Weather Conditions?", key="sim_custom_weather"):
        st.slider("Air Temperature (°C)", 10.0, 40.0, key="sim_air_temp")
        st.slider("Track Temperature (°C)", 15.0, 55.0, key="sim_track_temp")

with col2:
    st.subheader("Pit Stop Strategy")
    st.radio("Select Number of Pit Stops:", [0, 1, 2, 3], index=1, key="sim_num_stops", horizontal=True)
    st.caption("General guide: Softs: ~15-25 laps, Mediums: ~25-35, Hards: ~35-50.")
    stints, simulated_laps = [], 0
    for i in range(st.session_state.sim_num_stops + 1):
        st.markdown(f"**Stint {i+1}**")
        s_col1, s_col2 = st.columns(2)
        s_col1.selectbox(f"Tire Compound:", ["SOFT", "MEDIUM", "HARD"], key=f"c_{i}")
        s_col2.slider(f"Number of Laps:", 5, 50, key=f"l_{i}")
        stints.append({"compound": st.session_state[f"c_{i}"].lower(), "laps": st.session_state[f"l_{i}"]})
        simulated_laps += st.session_state[f"l_{i}"]
    st.metric("Strategy Laps / Official Laps", f"{simulated_laps} / {total_laps}", delta=f"{simulated_laps - total_laps} Laps")
    if simulated_laps != total_laps:
        st.warning("Strategy laps do not match official race distance.")

# --- Simulation Execution ---
if st.button("🏁 Run Simulation", use_container_width=True):
    simulate_api_url = f"{st.session_state.backend_address}/simulate"
    with st.spinner("Simulating race..."):
        payload = {
            "track": st.session_state.sim_track,
            "driver": st.session_state.sim_driver,
            "stints": stints
        }
        if st.session_state.sim_custom_weather:
            payload["air_temp"] = st.session_state.sim_air_temp
            payload["track_temp"] = st.session_state.sim_track_temp
        try:
            response = requests.post(simulate_api_url, json=payload)
            response.raise_for_status()
            results = response.json()

            if results:
                st.success("Simulation Complete!")

                summary = results.get("summary", {})
                lap_records = results.get("lap_records", [])

                st.subheader("Simulation Summary")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Laps Simulated", summary.get("total_laps_simulated", "N/A"))
                c2.metric("Avg Lap Time", f"{summary.get('average_lap_time', 0):.3f}s")
                best = summary.get("best_lap", {})
                c3.metric("Best Lap", f"{best.get('lap_time', 0):.3f}s", help=f"L{best.get('lap_number')} on {best.get('compound')}")
                worst = summary.get("worst_lap", {})
                c4.metric("Worst Lap", f"{worst.get('lap_time', 0):.3f}s", help=f"L{worst.get('lap_number')} on {worst.get('compound')}")
                st.metric("Predicted Total Race Time", f"{summary.get('total_race_time', 0):.3f}s")
                st.info(f"ℹ️ **Pit Stop Info:** {summary.get('pit_stop_info')}")

                st.subheader("Full Lap-by-Lap Data")
                if lap_records:
                    df = pd.DataFrame(lap_records)
                    st.dataframe(df, use_container_width=True)

                    st.subheader("Lap Time Evolution")
                    chart_df = df.copy()
                    pit_loss_match = re.search(r'adding ([\d\.]+)s', summary.get('pit_stop_info', ''))
                    if pit_loss_match and 'IsPitLap' in chart_df.columns:
                        pit_time = float(pit_loss_match.group(1))
                        pit_indices = chart_df[chart_df['IsPitLap']].index
                        chart_df.loc[pit_indices, 'LapTimeInSeconds'] -= pit_time

                    st.line_chart(chart_df, x="Lap Number", y="LapTimeInSeconds", color="Compound")

                # FIXED: Only transfer essential simulation data
                # Don't overwrite race engineer context values
                # Store simulation data for transfer
                st.session_state.sim_results = {
                    "stints": stints,
                    "race_time": f"{summary.get('total_race_time', 0):.3f}",
                    "best_lap": f"{best.get('lap_time', 0):.3f}",
                    "avg_lap": f"{summary.get('average_lap_time', 0):.3f}",
                    "worst_lap": f"{worst.get('lap_time', 0):.3f}"
                }
                
                # Store weather if custom
                if st.session_state.sim_custom_weather:
                    st.session_state.sim_results["air_temp"] = st.session_state.sim_air_temp
                    st.session_state.sim_results["track_temp"] = st.session_state.sim_track_temp
                
                st.session_state.simulation_results_exist = True
            

        except requests.exceptions.RequestException as e:
            st.error(f"API Error: Could not connect to the backend at {simulate_api_url}. Is it running? Details: {e}")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")

# --- Navigation to Race Engineer Page ---
if st.session_state.simulation_results_exist:
    st.divider()
    st.info("Want to discuss this strategy with the AI Race Engineer? Click the button below.")
    if st.button("💬 Discuss with Race Engineer", use_container_width=True):
        st.session_state.from_simulator = True
        st.switch_page("pages/2_💬_Race_Engineer.py")
