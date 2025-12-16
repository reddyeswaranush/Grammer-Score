import os
import glob
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import whisper
import torch
from transformers import BertTokenizer, BertModel

from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt


# set base paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE_DIR, "data")
TRAIN_AUDIO_DIR = os.path.join(DATA_DIR, "train_audio")
TEST_AUDIO_DIR  = os.path.join(DATA_DIR, "test_audio")

TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
TEST_CSV  = os.path.join(DATA_DIR, "test.csv")

# check if files and folders exist
print("BASE_DIR:", BASE_DIR)
print("TRAIN_AUDIO_DIR exists:", os.path.exists(TRAIN_AUDIO_DIR))
print("TEST_AUDIO_DIR exists :", os.path.exists(TEST_AUDIO_DIR))
print("TRAIN_CSV exists      :", os.path.exists(TRAIN_CSV))
print("TEST_CSV exists       :", os.path.exists(TEST_CSV))


# load csv files
print("\nLoading CSV files...")
train_df = pd.read_csv(TRAIN_CSV)
test_df  = pd.read_csv(TEST_CSV)

# clean filename column
train_df["filename"] = train_df["filename"].astype(str).str.strip()
test_df["filename"]  = test_df["filename"].astype(str).str.strip()

print("Train samples:", len(train_df))
print("Test samples :", len(test_df))


# function to safely find audio file
def resolve_audio_path(audio_dir, name):
    name = str(name).strip()

    # try direct match
    path = os.path.join(audio_dir, name)
    if os.path.exists(path):
        return path

    # try adding .wav
    path = path + ".wav"
    if os.path.exists(path):
        return path

    # try matching duplicates like _2.wav
    matches = glob.glob(os.path.join(audio_dir, name + "*.wav"))
    if matches:
        return matches[0]

    return None


# load whisper model
print("\nLoading Whisper model...")
whisper_model = whisper.load_model("base")


# convert audio to text
def audio_to_text(audio_path):
    try:
        result = whisper_model.transcribe(audio_path)
        return result["text"]
    except Exception as e:
        print("Error:", audio_path)
        return ""


# load bert model
print("\nLoading BERT model...")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertModel.from_pretrained("bert-base-uncased")
bert_model.eval()


# get bert embedding from text
def get_bert_embedding(text):
    if text.strip() == "":
        return np.zeros(768)

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    )

    with torch.no_grad():
        outputs = bert_model(**inputs)

    return outputs.last_hidden_state.mean(dim=1).squeeze().numpy()


# build training data
print("\nExtracting training features...")

X_embeddings = []
y_labels = []

for _, row in train_df.iterrows():

    audio_path = resolve_audio_path(TRAIN_AUDIO_DIR, row["filename"])

    if audio_path is None:
        print("Missing audio:", row["filename"])
        X_embeddings.append(np.zeros(768))
        y_labels.append(row["label"])
        continue

    text = audio_to_text(audio_path)
    embedding = get_bert_embedding(text)

    X_embeddings.append(embedding)
    y_labels.append(row["label"])

X = np.array(X_embeddings)
y = np.array(y_labels)


# split data
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# train regression model
print("\nTraining regression model...")
model = Ridge(alpha=1.0)
model.fit(X_train, y_train)


# calculate rmse
train_preds = model.predict(X_train)
val_preds   = model.predict(X_val)

train_rmse = np.sqrt(mean_squared_error(y_train, train_preds))
val_rmse   = np.sqrt(mean_squared_error(y_val, val_preds))

print("Training RMSE:", train_rmse)
print("Validation RMSE:", val_rmse)


# plot predictions vs actual
plt.scatter(y_val, val_preds)
plt.xlabel("True Score")
plt.ylabel("Predicted Score")
plt.title("Prediction vs Ground Truth")
plt.grid(True)
plt.show()


# predict test data
print("\nGenerating test predictions...")

test_embeddings = []

for _, row in test_df.iterrows():

    audio_path = resolve_audio_path(TEST_AUDIO_DIR, row["filename"])

    if audio_path is None:
        print("Missing test audio:", row["filename"])
        test_embeddings.append(np.zeros(768))
        continue

    text = audio_to_text(audio_path)
    embedding = get_bert_embedding(text)

    test_embeddings.append(embedding)

X_test = np.array(test_embeddings)
test_preds = model.predict(X_test)


# save submission file
submission = pd.DataFrame({
    "filename": test_df["filename"],
    "label": test_preds
})

submission.to_csv("submission.csv", index=False)

print("\nsubmission.csv saved")
print(submission.head())
