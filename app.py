import os
import sys
# Added render_template to the imports
from flask import Flask, request, jsonify, render_template 
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

app = Flask(__name__)

@app.route('/')
def index():
    # Changes the raw text string into a clean visual HTML layout rendering
    return render_template('index.html') 

@app.route('/predict', methods=['POST'])
def predict_datapoint():
    try:
        if request.is_json:
            data = request.get_json()
        else:
            data = request.form.to_dict()

        if not data:
            return jsonify({"error": "No input financial data payload received"}), 400

        numerical_data = {k: float(v) for k, v in data.items()}

        custom_data = CustomData(**numerical_data)
        pred_df = custom_data.get_data_as_data_frame()

        pipeline = PredictPipeline()
        results = pipeline.predict(pred_df)
        
        prediction_output = int(results[0])
        status = "Bankrupt / At High Financial Risk" if prediction_output == 1 else "Stable / Solvent / Low Risk"

        return jsonify({
            "prediction_code": prediction_output,
            "status": status
        })

    except Exception as e:
        return jsonify({"error": f"Prediction failed inside web engine layer: {str(e)}"}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)