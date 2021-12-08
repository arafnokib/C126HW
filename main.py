from flask import Flask, json, jsonify, request
from classifier import get_prediction
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
@app.route('/predict-abc', methods=["POST"])
def predictABC():
    image = request.files.get("alphabet")
    prediction = get_prediction(image)
    
    return jsonify({"prediction" : prediction})


if(__name__ == "__main__"):
    app.run(debug=True)