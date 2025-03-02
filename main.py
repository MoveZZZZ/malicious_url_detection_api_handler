from flask import Flask, request, jsonify
from flask_cors import CORS
import time
from Ensamble import ModelEnsembler




app = Flask(__name__)
CORS(app)

_Ensambler = ModelEnsembler()

@app.route('/api/data', methods=['POST'])
def handle_request():
    try:
        data = request.json
        url = data.get("url", None)
        print(f"\nURL: {url}")

        if not url:
            return jsonify({"error": "No URL provided"}), 400

        time_start = time.time()
        predicted_label, chanse = _Ensambler.predict_url(url)
        print(f"Label: {predicted_label}, chanse: {chanse}")
        time_end = time.time()
        print(f"Time to process URL: {time_end - time_start}")
        return jsonify({
            "predicted_label": predicted_label,
            "prediction_probs": round(chanse, 2)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

