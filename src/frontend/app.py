import streamlit as st
import requests
import pandas as pd
import re

# --- Configuration ---
FLASK_API_URL = "https://lamaq-f1-strategy-api.hf.space/simulate" # URL of our running Flask API

# --- Data for UI ---
# [UPDATED] Dictionary mapping tracks to their official 2025 race lap counts
TRACK_DATA = {
    "Melbourne": 58,
    "Shanghai": 56,
    "Suzuka": 53,
    "Bahrain": 57,
    "Jeddah": 50,
    "Miami": 57,
    "Imola": 63,
    "Monaco": 78,
    "Circuit de Barcelona-Catalunya": 66,
    "Montreal": 70,
    "Spielberg": 71,
    "Silverstone": 52,
    "Spa-Francorchamps": 44,
    "Budapest": 70,
    "Zandvoort": 72,
    "Monza": 53,
    "Baku": 51,
    "Singapore": 62,
    "Austin": 56,
    "Mexico City": 71,
    "Sao Paulo": 71,
    "Las Vegas": 50,
    "Lusail": 57,
    "Yas Marina": 58
}
TRACK_LIST = sorted(TRACK_DATA.keys())

# Updated with the confirmed 2025 F1 driver lineup
DRIVER_LIST = ['VER', 'DOO', 'ANT', 'PIA', 'RUS', 'ALB', 'NOR', 'SAI', 'LEC',
       'TSU', 'OCO', 'HAD', 'HAM', 'BOR', 'HUL', 'STR', 'BEA', 'ALO',
       'GAS', 'LAW', 'COL']


# --- UI Layout ---
st.set_page_config(layout="wide")
st.title("🏎️ F1 Race Strategy Simulator")
st.markdown("Design a pit stop strategy and simulate a race to see the predicted lap times based on our trained tire degradation model.")

# --- User Inputs ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("Race Setup")
    track = st.selectbox("Select Track:", TRACK_LIST)
    driver = st.selectbox("Select Driver:", sorted(DRIVER_LIST))
    
    # [UPDATED] Dynamically set the total laps based on the selected track
    total_laps = TRACK_DATA.get(track, 55)
    st.metric("Official Race Laps", total_laps)
    
    use_custom_temps = st.checkbox("Set Custom Weather Conditions?")
    air_temp = None
    track_temp = None
    if use_custom_temps:
        air_temp = st.slider("Air Temperature (°C)", 10.0, 40.0, 25.0)
        track_temp = st.slider("Track Temperature (°C)", 15.0, 55.0, 35.0)

with col2:
    st.subheader("Pit Stop Strategy")
    num_stops = st.radio("Select Number of Pit Stops:", [1, 2, 3], index=1, horizontal=True)

    # Disclaimer text for tire lifespan
    tyre_disclaimer = """
    **Disclaimer:** Actual tyre life varies. General guide:
    - **Soft:** ~15-25 laps
    - **Medium:** ~25-35 laps
    - **Hard:** ~35-50 laps
    """
    st.caption(tyre_disclaimer)
    
    stints = []
    simulated_laps = 0
    for i in range(num_stops + 1):
        st.markdown(f"**Stint {i+1}**")
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            compound = st.selectbox(f"Tire Compound:", ["SOFT", "MEDIUM", "HARD"], key=f"compound_{i}", index=(i % 3))
        with col_s2:
            laps = st.slider(f"Number of Laps:", 5, 40, 20, key=f"laps_{i}")


        stints.append({"compound": compound.lower(), "laps": laps})
        simulated_laps += laps
    
    # --- THE FIX ---
    # Show a dynamic metric for the user's strategy laps vs official laps
    st.metric("Strategy Laps / Official Laps", f"{simulated_laps} / {total_laps}")
    # Show a small caption only if the laps do not match
    if simulated_laps != total_laps:
        st.caption("⚠️ Strategy laps do not match official race distance.")

# --- Simulation Trigger ---
if st.button("🏁 Run Simulation", use_container_width=True):
    with st.spinner("Simulating race..."):
        # 1. Construct the JSON payload for the API
        payload = {
            "track": track,
            "driver": driver,
            "stints": stints,
        }
        if air_temp is not None:
            payload["air_temp"] = air_temp
        if track_temp is not None:
            payload["track_temp"] = track_temp
            
        try:
            # 2. Make the POST request to the Flask API
            response = requests.post(FLASK_API_URL, json=payload)
            response.raise_for_status()
            
            results = response.json()
            
            # 3. Display the results
            st.success("Simulation Complete!")
            
            summary = results.get("summary", {})
            lap_records = results.get("lap_records", [])
            
            st.subheader("Simulation Summary")
            
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Laps Simulated", summary.get("total_laps_simulated", "N/A"))
            c2.metric("Average Lap Time", f"{summary.get('average_lap_time', 0):.3f}s")
            
            best = summary.get("best_lap", {})
            c3.metric("Best Lap Time", f"{best.get('lap_time', 0):.3f}s", help=f"Lap {best.get('lap_number')} on {best.get('compound')} tires")
            
            worst = summary.get("worst_lap", {})
            c4.metric("Worst Lap Time", f"{worst.get('lap_time', 0):.3f}s", help=f"Lap {worst.get('lap_number')} on {worst.get('compound')} tires")

            st.metric("Predicted Total Race Time", f"{summary.get('total_race_time', 0):.3f}s")
            st.info(f"ℹ️ **Pit Stop Info:** {summary.get('pit_stop_info')}")
            
            st.subheader("Full Lap-by-Lap Data")
            if lap_records:
                df = pd.DataFrame(lap_records)
                
                # --- Charting Logic with Pit Stop Spikes ---
                chart_df = df.copy()
                pit_stop_info = summary.get('pit_stop_info', "")
                
                match = re.search(r'adding ([\d\.]+)s each', pit_stop_info)
                if match:
                    pit_stop_time = float(match.group(1))
                    
                    pit_lap_number = 0
                    for stint in stints[:-1]:
                        pit_lap_number += stint['laps']
                        if pit_lap_number < len(chart_df):
                           chart_df.loc[pit_lap_number, 'LapTimeInSeconds'] += pit_stop_time

                st.dataframe(df, use_container_width=True)
                
                st.subheader("Lap Time Evolution (including pit stop time loss)")
                st.line_chart(chart_df, x="Lap Number", y="LapTimeInSeconds", color="Compound")

        except requests.exceptions.RequestException as e:
            st.error(f"API Connection Error: Could not connect to the simulation backend. Is it running?")
            st.code(e)
        except Exception as e:
            st.error(f"An unexpected error occurred.")
            st.code(e)
