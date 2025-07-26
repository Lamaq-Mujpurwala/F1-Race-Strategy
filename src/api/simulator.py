import pandas as pd
from model_loader import model_loader # We import the loader instance

# A simple mapping for default temperatures if the user doesn't provide them.
DEFAULT_TRACK_TEMPS = {
    "Silverstone": {"airtemp": 20.0, "tracktemp": 30.0},
    "Monza": {"airtemp": 25.0, "tracktemp": 40.0},
    "Bahrain": {"airtemp": 28.0, "tracktemp": 35.0},
    "Spa-Francorchamps": {"airtemp": 18.0, "tracktemp": 25.0},
    # Add other tracks as needed
}

class RaceSimulator:
    def __init__(self, model, preprocessor):
        """
        Initializes the simulator with a loaded model and preprocessor.
        """
        self.model = model
        self.preprocessor = preprocessor

    def _predict_lap_time(self, lap_data: dict) -> float:
        """
        Predicts a single lap's time using the loaded model and preprocessor.
        """
        if not self.model or not self.preprocessor:
            return 95.0 

        df = pd.DataFrame([lap_data])
        transformed_data = self.preprocessor.transform(df)
        prediction = self.model.predict(transformed_data)
        return float(prediction[0])

    def run_simulation(self, strategy: dict) -> dict:
        """
        Runs a full race simulation based on a user-defined strategy.
        """
        # --- 1. Extract and validate inputs (using lowercase) ---
        base_params = {
            "track": strategy.get("track"),
            "driver": strategy.get("driver"),
            "year": 2025,
        }
        
        default_temps = DEFAULT_TRACK_TEMPS.get(base_params["track"], {"airtemp": 22.0, "tracktemp": 32.0})
        base_params["airtemp"] = strategy.get("air_temp", default_temps["airtemp"])
        base_params["tracktemp"] = strategy.get("track_temp", default_temps["tracktemp"])

        stints = strategy.get("stints", [])
        if not stints:
            raise ValueError("Strategy must include at least one stint.")

        # --- 2. Run the lap-by-lap simulation ---
        lap_records = []
        total_laps = sum(stint['laps'] for stint in stints)
        current_lap_number = 1

        for stint_info in stints:
            compound = stint_info['compound']
            num_laps_in_stint = stint_info['laps']
            
            for tyre_lap in range(1, num_laps_in_stint + 1):
                # Using lowercase keys to match the training data columns
                lap_data = {
                    "tyrelife": float(tyre_lap),
                    "lapnumber": float(current_lap_number),
                    **base_params,
                    "compound": compound,
                }
                
                predicted_time = self._predict_lap_time(lap_data)
                
                lap_records.append({
                    "Lap Number": current_lap_number,
                    "Compound": compound,
                    "TyreLife": tyre_lap,
                    "LapTimeInSeconds": round(predicted_time, 3)
                })
                current_lap_number += 1

        # --- 3. Calculate summary statistics ---
        if not lap_records:
            return {"error": "Simulation produced no results."}

        results_df = pd.DataFrame(lap_records)
        best_lap = results_df.loc[results_df['LapTimeInSeconds'].idxmin()]
        worst_lap = results_df.loc[results_df['LapTimeInSeconds'].idxmax()]
        
        summary = {
            "total_laps_simulated": total_laps,
            "average_lap_time": round(results_df['LapTimeInSeconds'].mean(), 3),
            "best_lap": {
                "lap_time": best_lap['LapTimeInSeconds'],
                "lap_number": int(best_lap['Lap Number']),
                "compound": best_lap['Compound'],
            },
            "worst_lap": {
                "lap_time": worst_lap['LapTimeInSeconds'],
                "lap_number": int(worst_lap['Lap Number']),
                "compound": worst_lap['Compound'],
            },
            "pit_stop_disclaimer": "Total race time does not include time lost during pit stops (typically +2.0-5.0s per stop)."
        }

        return {
            "summary": summary,
            "lap_records": results_df.to_dict(orient='records')
        }

# Create a single instance of the simulator, passing the loaded model and preprocessor
simulator = RaceSimulator(model=model_loader.model, preprocessor=model_loader.preprocessor)
