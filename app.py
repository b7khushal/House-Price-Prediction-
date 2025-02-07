from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load model and transformers
model = pickle.load(open("model.pkl", "rb"))
encoder = pickle.load(open("encoder.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.form.to_dict()

        # Convert inputs
        numerical_values = np.array([
            float(data["INT_SQFT"]),
            float(data["DIST_MAINROAD"]),
            float(data["N_BEDROOM"]),
            float(data["N_BATHROOM"]),
            float(data["N_ROOM"]),
        ]).reshape(1, -1)

        categorical_values = np.array([[data["AREA"], data["SALE_COND"], data["PARK_FACIL"]]])

        # Transform inputs
        scaled_numerical = scaler.transform(numerical_values)
        encoded_categorical = encoder.transform(categorical_values)

        # Combine and predict
        final_features = np.hstack((scaled_numerical, encoded_categorical))
        prediction = model.predict(final_features)[0]

        return render_template("index.html", prediction_text=f"Predicted House Price: ₹{int(prediction)}")
    
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
