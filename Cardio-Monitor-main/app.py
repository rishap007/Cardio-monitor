from pathlib import Path

from flask import Flask, render_template, request
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import prediction
import visualization


app = Flask(__name__)
STATIC_DIR = Path(__file__).resolve().parent / "static"
counter = 0
counter2 = 0


def _save_comparison(data, labels, filename):
    normal, user = data
    width = 0.25
    positions = np.arange(len(normal))
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.bar(positions, normal, color="green", width=width, edgecolor="grey", label="Typical value")
    ax.bar(positions + width, user, color="red", width=width, edgecolor="grey", label="Your value")
    ax.set_xlabel("Health status attributes", fontweight="bold", fontsize=15)
    ax.set_ylabel("Values", fontweight="bold", fontsize=15)
    ax.set_xticks(positions + width / 2)
    ax.set_xticklabels(labels)
    ax.legend()
    fig.tight_layout()
    fig.savefig(STATIC_DIR / filename)
    plt.close(fig)


@app.route("/")
def home():
    global counter2
    counter2 += 1
    return render_template("home.html", all_count=counter2)


@app.route("/predict", methods=["POST"])
def predict():
    global counter
    fields = (
        "age", "sex", "cp", "trestbps", "restecg", "chol", "fbs",
        "thalach", "exang", "oldpeak", "slope", "ca", "thal",
    )
    values = {field: request.form.get(field, "") for field in fields}
    try:
        result = prediction.preprocess(**values)
    except ValueError as error:
        return render_template("error.html", message=str(error)), 400

    counter = counter + 1 if counter < 50 else 0
    data1, data2 = visualization.visualizationpreprocess(
        *(values[field] for field in fields), result
    )
    _save_comparison(data1, ["cp", "fbs", "restecg", "exang", "oldpeak", "slope", "ca", "thal"], "plotng.png")
    _save_comparison(data2, ["trestbps", "chol", "thalach"], "plotng2.png")
    patient_name = request.form.get("name", "").strip() or "there"
    return render_template(
        "result.html", prediction=result, nameofpatient=patient_name,
        model_counter=counter, total_counter=counter2,
    )


@app.route("/about")
def about():
    return render_template("disease.html")


@app.errorhandler(500)
def internal_error(error):
    return render_template("error.html", message="Something went wrong. Please try again."), 500


@app.errorhandler(404)
def not_found(error):
    return "404 error", 404


if __name__ == "__main__":
    app.run(debug=False)
