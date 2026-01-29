from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
import boto3
import os
import numpy as np
import joblib
import pandas as pd
from dotenv import load_dotenv
from scipy import sparse

load_dotenv()

SEED = 42
np.random.seed(SEED)

TRAIN_PATH = "data/train.parquet.snappy"
ARTIFACT_DIR = "artifacts"
os.makedirs(ARTIFACT_DIR, exist_ok=True)

CHUNK_SIZE = 5000
TEXT_COLS = ["name", "description", "vendor", "model", "type_prefix"]
VAL_SIZE = 20000
MAX_FEATURES = 20000
NGRAM_WORD = (1, 2)
NGRAM_CHAR = (3, 5)
MIN_DF = 2

bucket = "k.kupitman"
prefix = "project_MLOps/artifacts/"

s3 = boto3.client(
    "s3",
    endpoint_url="https://minio.v-efimov.tech",
    aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
    aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
    region_name='us-east-1'
)

def build_text(df):
    return df[TEXT_COLS].fillna("").astype(str).agg(" ".join, axis=1).str.lower()

if not os.path.exists(TRAIN_PATH):
    os.makedirs(os.path.dirname(TRAIN_PATH), exist_ok=True)
    s3.download_file(bucket, "project_MLOps/data/train.parquet.snappy", TRAIN_PATH)

train_full = pd.read_parquet(TRAIN_PATH)

sample = train_full.sample(n=20000, random_state=SEED)

tfidf_word = TfidfVectorizer(max_features=MAX_FEATURES//2, ngram_range=NGRAM_WORD, min_df=MIN_DF)
tfidf_char = TfidfVectorizer(max_features=MAX_FEATURES//2, ngram_range=NGRAM_CHAR, analyzer='char_wb', min_df=MIN_DF)

tfidf_word.fit(build_text(sample))
tfidf_char.fit(build_text(sample))

joblib.dump(tfidf_word, f"{ARTIFACT_DIR}/tfidf_word.pkl")
joblib.dump(tfidf_char, f"{ARTIFACT_DIR}/tfidf_char.pkl")

val = train_full.sample(n=VAL_SIZE, random_state=SEED+1)
train_chunks = [train_full.drop(val.index).iloc[i:i+CHUNK_SIZE] for i in range(0, len(train_full) - VAL_SIZE, CHUNK_SIZE)]

X_val_word = tfidf_word.transform(build_text(val))
X_val_char = tfidf_char.transform(build_text(val))
X_val = sparse.hstack([X_val_word, X_val_char])
y_val = val["category_ind"]

all_classes = train_full["category_ind"].unique()
model = SGDClassifier(loss="hinge", max_iter=1, tol=None, random_state=SEED)

best_model = None
best_acc = 0.0

for i, chunk in enumerate(train_chunks):
    X_chunk_word = tfidf_word.transform(build_text(chunk))
    X_chunk_char = tfidf_char.transform(build_text(chunk))
    X_chunk = sparse.hstack([X_chunk_word, X_chunk_char])
    y_chunk = chunk["category_ind"]

    model.partial_fit(X_chunk, y_chunk, classes=all_classes)

    y_pred_val = model.predict(X_val)
    acc = accuracy_score(y_val, y_pred_val)

    if acc > best_acc:
        best_acc = acc
        joblib.dump(model, f"{ARTIFACT_DIR}/model.pkl")

print(f"Лучшая accuracy: {best_acc:.4f}")

files_to_upload = [
    ("model.pkl", "model.pkl"),
    ("tfidf_word.pkl", "tfidf_word.pkl"),
    ("tfidf_char.pkl", "tfidf_char.pkl")
]

for local_file, s3_name in files_to_upload:
    local_path = f"{ARTIFACT_DIR}/{local_file}"
    if os.path.exists(local_path):
        s3.upload_file(local_path, bucket, prefix + s3_name)