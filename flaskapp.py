from flask import Flask, request, render_template_string
import numpy as np
import pandas as pd
import joblib
import os

# Define the path to the saved model
# Make sure 'random_forest_model.pkl' is in the same directory as your Flask app or provide the full path
MODEL_PATH = 'random_forest_model.pkl'

# Check if the model file exists
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}. Please ensure the model is saved correctly.")

# Load the trained Random Forest model
model = joblib.load(MODEL_PATH)

app = Flask(__name__)

# Define the HTML template for the form and results
HTML_TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
    <title>Smart Fetal Health Predictor</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body {
            background: #f0f9ff;
            font-family: 'Segoe UI', sans-serif;
        }
        .container {
            max-width: 720px;
            margin-top: 50px;
        }
        .card {
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
            border: none;
        }
        .card-header {
            background: #0066cc;
            color: white;
            font-weight: bold;
        }
        .btn-primary {
            background-color: #0066cc;
            border: none;
        }
        .result {
            font-size: 1.25rem;
            font-weight: bold;
        }
    </style>
</head>
<body>

<div class="container">
    <div class="card">
        <div class="card-header text-center">
            🩺 Smart Fetal Health Predictor
        </div>
        <div class="card-body">
            <form method="post" class="row g-3">
                {% for key, value in {
                    "abnormal_short_term_variability": "Abnormal Short Term Variability",
                    "mean_value_of_short_term_variability": "Mean Value of STV",
                    "percentage_of_time_with_abnormal_long_term_variability": "% Abnormal LTV",
                    "histogram_mean": "Histogram Mean",
                    "histogram_mode": "Histogram Mode",
                    "histogram_median": "Histogram Median",
                    "prolongued_decelerations": "Prolongued Decelerations",
                    "mean_value_of_long_term_variability": "Mean LTV",
                    "accelerations": "Accelerations",
                    "histogram_variance": "Histogram Variance"
                }.items() %}
                <div class="col-md-6">
                    <label class="form-label">{{ value }}</label> 
                    <input type="number" step="0.01" name="{{ key }}" value="{{ form_values.get(key, '') }}" class="form-control" required>
                </div>
                {% endfor %}
                <div class="col-12 text-center">
                    <button type="submit" class="btn btn-primary btn-lg">🔍 Predict</button>
                </div>
            </form>

            {% if prediction %}
            <hr>
            <div class="alert alert-info text-center mt-4 result">
                <span>🧠 Fetal Health Status: <strong>{{ prediction }}</strong></span>
            </div>
            {% endif %}
        </div>
    </div>
</div>

</body>
</html>
"""



# Define the route for the home page and prediction
@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    # Initialize form_values with None or empty strings
    form_values = {
        "abnormal_short_term_variability": None,
        "mean_value_of_short_term_variability": None,
        "percentage_of_time_with_abnormal_long_term_variability": None,
        "histogram_mean": None,
        "histogram_mode": None,
        "histogram_median": None,
        "prolongued_decelerations": None,
        "mean_value_of_long_term_variability": None,
        "accelerations": None,
        "histogram_variance": None
    }

    if request.method == 'POST':
        try:
            # Get data from the form
            input_data = {
                "abnormal_short_term_variability": float(request.form['abnormal_short_term_variability']),
                "mean_value_of_short_term_variability": float(request.form['mean_value_of_short_term_variability']),
                "percentage_of_time_with_abnormal_long_term_variability": float(request.form['percentage_of_time_with_abnormal_long_term_variability']),
                "histogram_mean": float(request.form['histogram_mean']),
                "histogram_mode": float(request.form['histogram_mode']),
                "histogram_median": float(request.form['histogram_median']),
                "prolongued_decelerations": float(request.form['prolongued_decelerations']),
                "mean_value_of_long_term_variability": float(request.form['mean_value_of_long_term_variability']),
                "accelerations": float(request.form['accelerations']),
                "histogram_variance": float(request.form['histogram_variance'])
            }

            # Create a pandas DataFrame from the input data
            input_df = pd.DataFrame([input_data])

            # Make a prediction
            prediction_value = model.predict(input_df)[0]

            # Map the prediction value to a label
            prediction = {
                1.0: "🟢 Normal",
                2.0: "🟠 Suspect",
                3.0: "🔴 Pathological"
            }.get(prediction_value, "Unknown")

            # Keep the submitted values in the form
            form_values.update(input_data)

        except Exception as e:
            prediction = f"Error: {e}"

    # Render the HTML template with or without the prediction result
    #return render_template_string(HTML_TEMPLATE, prediction=prediction, **form_values)
    #return render_template_string(HTML_TEMPLATE, prediction=prediction, form_values=form_values)
    return render_template_string(HTML_TEMPLATE, prediction=prediction, form_values=form_values)



if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')