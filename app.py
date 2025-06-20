import numpy as np

import pandas as pd
from sklearn.preprocessing import StandardScaler
from flask import Flask, request, jsonify, render_template
from src.pipelines.prediction_pipeline import CustomData, PridictionPipeline

application = Flask(__name__)

app = application  # For compatibility with WSGI servers

## Routes


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict_datapoint", methods=["POST", "GET"])
def predict_datapoint():
    if request.method == "GET":
        return render_template("home.html")
    else:
        try:
            data = CustomData(
                gender=request.form.get("gender"),
                race_ethnicity=request.form.get("ethnicity"),
                parental_level_of_education=request.form.get(
                    "parental_level_of_education"
                ),
                lunch=request.form.get("lunch"),
                test_preparation_course=request.form.get("test_preparation_course"),
                reading_score=float(request.form.get("writing_score")),
                writing_score=float(request.form.get("reading_score")),
            )
            pred_df = data.get_data_as_dataframe()
            print(pred_df)
            print("Before Prediction")

            predict_pipeline = PridictionPipeline()
            print("Mid Prediction")
            results = predict_pipeline.predict(pred_df)
            print("after Prediction")
            return render_template("home.html", results=results[0])

        except Exception as e:
            return jsonify({"error": str(e)})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
