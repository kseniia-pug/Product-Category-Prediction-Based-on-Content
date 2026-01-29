import pandas as pd
import joblib
import os
import numpy as np
import boto3

SEED = 42
np.random.seed(SEED)

TEST_PATH = "data/test.parquet.snappy"
ARTIFACT_DIR = "artifacts"
CHUNK_SIZE = 5000
TEXT_COLS = ["name", "description", "vendor", "model", "type_prefix"]

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

def build_text(df):
    return df[TEXT_COLS].fillna("").astype(str).agg(" ".join, axis=1).str.lower()

tfidf_word = joblib.load(f"{ARTIFACT_DIR}/tfidf_word.pkl")
tfidf_char = joblib.load(f"{ARTIFACT_DIR}/tfidf_char.pkl")
model = joblib.load(f"{ARTIFACT_DIR}/model.pkl")

if not os.path.exists(TEST_PATH):
    os.makedirs(os.path.dirname(TEST_PATH), exist_ok=True)
    s3.download_file(bucket, "project_MLOps/data/test.parquet.snappy", TEST_PATH)

test_full = pd.read_parquet(TEST_PATH)

preds = []
for i in range(0, len(test_full), CHUNK_SIZE):
    chunk = test_full.iloc[i:i+CHUNK_SIZE]
    X_chunk_word = tfidf_word.transform(build_text(chunk))
    X_chunk_char = tfidf_char.transform(build_text(chunk))
    X_chunk = np.hstack([X_chunk_word.toarray(), X_chunk_char.toarray()])
    preds_chunk = model.predict(X_chunk)
    preds.extend(preds_chunk)

submission = pd.DataFrame({
    "ID": np.arange(0, len(preds)),
    "category_ind": preds
})
submission.to_csv("submission.csv", index=False)