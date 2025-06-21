from flask import Flask, render_template, request
import pickle
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

# Load model and tokenizer
model = load_model("best_model.keras")
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

label_map = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    if request.method == "POST":
        text = request.form["text"]
        seq = tokenizer.texts_to_sequences([text])
        padded = pad_sequences(seq, maxlen=50, padding='post')
        pred = np.argmax(model.predict(padded), axis=-1)[0]
        result = label_map[pred]
    return render_template("index.html", result=result)


if __name__ == "__main__":
    app.run(debug=True)
