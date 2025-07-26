from flask import Flask, jsonify, request
import os
from simulator import simulator # Import the simulator instance
# Create the Flask application object
app = Flask(__name__)

@app.route("/health", methods=["GET"])
def health_check():
    """
    A simple endpoint to confirm that the API is running and the model is loaded.
    """
    if simulator.model and simulator.preprocessor:
        return jsonify({"status": "ok", "message": "API is healthy and model is loaded."})
    else:
        return jsonify({"status": "error", "message": "API is running, but model failed to load."}), 500

@app.route("/simulate", methods=["POST"])
def simulate_strategy():
    """
    The main endpoint to run a race simulation.
    Expects a JSON payload with the strategy details.
    """
    if not simulator.model:
        return jsonify({"error": "Model is not loaded. Cannot run simulation."}), 503

    strategy_params = request.get_json()
    if not strategy_params:
        return jsonify({"error": "Missing JSON request body."}), 400

    # Basic validation using lowercase keys
    required_keys = ["track", "driver", "stints"]
    if not all(key in strategy_params for key in required_keys):
        return jsonify({"error": f"Request must include {required_keys}."}), 400

    try:
        results = simulator.run_simulation(strategy_params)
        return jsonify(results)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"An unexpected error occurred: {e}"}), 500

# This block allows us to run the app directly for local testing
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)
