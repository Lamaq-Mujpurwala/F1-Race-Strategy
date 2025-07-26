import streamlit as st
import requests
import pandas as pd

# --- Configuration ---
FLASK_API_URL = "http://127.0.0.1:8080/simulate" # URL of our running Flask API

# --- UI Layout ---
st.set_page_config(layout="wide")
st.title("🏎️ F1 Race Strategy Simulator")

# --- User Inputs ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("Race Setup")
    track = st.selectbox("Select Track:", ["Silverstone", "Monza", "Bahrain", "Spa-Francorchamps"])
    driver = st.selectbox("Select Driver:", ["VER", "HAM", "LEC", "NOR", "PIA"])
    
    use_custom_temps = st.checkbox("Set Custom Weather Conditions?")
    air_temp = None
    track_temp = None
    if use_custom_temps:
        air_temp = st.slider("Air Temperature (°C)", 10.0, 40.0, 25.0)
        track_temp = st.slider("Track Temperature (°C)", 15.0, 55.0, 35.0)

with col2:
    st.subheader("Pit Stop Strategy")
    num_stops = st.radio("Select Number of Pit Stops:", [1, 2, 3], index=1, horizontal=True)
    
    stints = []
    total_laps = 0
    for i in range(num_stops + 1):
        st.markdown(f"**Stint {i+1}**")
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            compound = st.selectbox(f"Tire Compound:", ["SOFT", "MEDIUM", "HARD"], key=f"compound_{i}")
        with col_s2:
            laps = st.slider(f"Number of Laps:", 5, 35, 20, key=f"laps_{i}")
        
        stints.append({"compound": compound, "laps": laps})
        total_laps += laps
    
    st.metric("Total Race Laps", total_laps)

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
            response.raise_for_status() # Raise an exception for bad status codes
            
            results = response.json()
            
            # 3. Display the results
            st.success("Simulation Complete!")
            
            summary = results.get("summary", {})
            lap_records = results.get("lap_records", [])
            
            st.subheader("Simulation Summary")
            
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Laps", summary.get("total_laps_simulated", "N/A"))
            c2.metric("Average Lap Time", f"{summary.get('average_lap_time', 0):.3f}s")
            
            best = summary.get("best_lap", {})
            c3.metric("Best Lap Time", f"{best.get('lap_time', 0):.3f}s", help=f"Lap {best.get('lap_number')} on {best.get('compound')} tires")
            
            worst = summary.get("worst_lap", {})
            c4.metric("Worst Lap Time", f"{worst.get('lap_time', 0):.3f}s", help=f"Lap {worst.get('lap_number')} on {worst.get('compound')} tires")

            st.info(f"ℹ️ **Disclaimer:** {summary.get('pit_stop_disclaimer')}")
            
            st.subheader("Full Lap-by-Lap Data")
            if lap_records:
                df = pd.DataFrame(lap_records)
                st.dataframe(df, use_container_width=True)
                
                # Chart
                st.line_chart(df, x="Lap Number", y="LapTimeInSeconds", color="Compound")

        except requests.exceptions.RequestException as e:
            st.error(f"API Connection Error: Could not connect to the simulation backend. Is it running?")
            st.code(e)
        except Exception as e:
            st.error(f"An unexpected error occurred.")
            st.code(e)
