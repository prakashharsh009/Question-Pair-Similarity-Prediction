from main import predict
import warnings

warnings.filterwarnings("ignore")
from flask import Flask, jsonify, request, redirect, url_for, render_template

q1 = "What is the best way to get a good laptop?"
q2 = "What is the best way to get a laptop?"
prob = "yes"


app = Flask(__name__)


@app.route("/")
def inputs():
    return render_template("index.html")


@app.route("/output/", methods=["POST"])
def output():
    a = "me"
    data = request.form.to_dict()
    q1 = data.get("q1")
    q2 = data.get("q2")
    prob = data.get("probabiliy")

    y_q, y_q_proba = predict(q1, q2, prob)

    result = dict()
    result["Question-1"] = q1
    result["Question-2"] = q2
    if y_q == 1:
        result["Predicted Class"] = "Similar"
    else:
        result["Predicted Class"] = "Not Similar"

    if prob == "yes":
        result["Probabiliy"] = round(max(y_q_proba[0]), 4)

    return render_template("output.html", result=result)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=False)
