from flask import Flask, request, jsonify
import joblib
import os
import numpy as np
import pandas as pd
import boto3
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

ARTIFACT_DIR = "artifacts"
TEXT_COLS = ["name", "description", "vendor", "model", "type_prefix"]
os.makedirs(ARTIFACT_DIR, exist_ok=True)

bucket = "k.kupitman"
prefix = "project_MLOps/artifacts/"

s3 = boto3.client(
    "s3",
    endpoint_url="https://minio.v-efimov.tech",
    aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
    aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
    region_name='us-east-1'
)

files_to_download = [
    ("model.pkl", "model.pkl"),
    ("tfidf_word.pkl", "tfidf_word.pkl"),
    ("tfidf_char.pkl", "tfidf_char.pkl")
]

for local_file, s3_file in files_to_download:
    local_path = f"{ARTIFACT_DIR}/{local_file}"
    s3.download_file(bucket, prefix + s3_file, local_path)

tfidf_word = joblib.load(f"{ARTIFACT_DIR}/tfidf_word.pkl")
tfidf_char = joblib.load(f"{ARTIFACT_DIR}/tfidf_char.pkl")
model = joblib.load(f"{ARTIFACT_DIR}/model.pkl")

def build_text(df):
    return df[TEXT_COLS].fillna("").astype(str).agg(" ".join, axis=1).str.lower()

def make_prediction(data_item):
    texts = data_item.get('texts', {})
    row_data = {col: texts.get(col, "") for col in TEXT_COLS}
    texts_df = pd.DataFrame([row_data])
    combined_text = build_text(texts_df)
    X_word = tfidf_word.transform(combined_text)
    X_char = tfidf_char.transform(combined_text)
    X = np.hstack([X_word.toarray(), X_char.toarray()])
    prediction = model.predict(X)
    return prediction[0]

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok"})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        if not data:
            return jsonify({"error": "No data provided"}), 400
        if isinstance(data, dict):
            data = [data]

        predictions = []
        for item in data:
            if 'texts' not in item:
                return jsonify({"error": "Missing 'texts' field in one of the items"}), 400
            pred = make_prediction(item)
            predictions.append({"category_ind": int(pred)})
        if len(predictions) == 1:
            return jsonify(predictions[0])
        else:
            return jsonify(predictions)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('SERVICE_PORT', 8000))
    app.run(host='0.0.0.0', port=port)
