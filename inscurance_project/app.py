from flask import Flask, render_template, request
import joblib
import numpy as np

# Load trained model
model = joblib.load("insurance_model.lb")  # Apna model yahi save karna

app = Flask(__name__)

# Routes
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/contact")
def contact():
    return render_template("contact.html")

@app.route("/project", methods=["POST", "GET"])
def project():
    if request.method == "POST":
        # Form data from HTML
        age = int(request.form["age"])
        sex = int(request.form["sex"])
        bmi = float(request.form["bmi"])
        children = int(request.form["children"])
        smoker = int(request.form["smoker"])
        region = int(request.form["region"])

        # Prepare input for model
        print('output>>>>>>',age, sex, bmi, children, smoker, region)
        data = np.array([[age, sex, bmi, children, smoker, region]])
        prediction = model.predict(data).item()
        prediction = round(prediction, 2)

        print("Prediction from model:", prediction)
        return render_template("project.html", prediction=prediction)

    return render_template("project.html")

# Run the app
if __name__ == "__main__":
    app.run(debug=True)
