from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load model and vectorizer
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

with open("vectorizer.pkl", "rb") as file:
    vectorizer = pickle.load(file)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None

    if request.method == "POST":
        news_text = request.form["news"]

        transformed_text = vectorizer.transform([news_text])
        result = model.predict(transformed_text)[0]

        if result == 1:
            prediction = "Fake News ❌"
        else:
            prediction = "Real News ✅"

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
