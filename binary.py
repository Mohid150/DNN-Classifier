
# Deepfake Audio Detection Using MFCC and Classical + Deep Learning Models

# === STEP 1: IMPORT LIBRARIES ===
import os
import librosa
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.svm import SVC
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import joblib

# === STEP 2: DEFINE FEATURE EXTRACTION FUNCTION ===
def extract_features_recursive(root_dir, label_value, sr=16000, n_mfcc=13, max_len=200):
    features, labels = [], []
    for dirpath, _, filenames in os.walk(root_dir):
        for file in filenames:
            if file.endswith(".wav"):
                try:
                    file_path = os.path.join(dirpath, file)
                    signal, _ = librosa.load(file_path, sr=sr)
                    mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=n_mfcc)
                    if mfcc.shape[1] < max_len:
                        pad_width = max_len - mfcc.shape[1]
                        mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
                    else:
                        mfcc = mfcc[:, :max_len]
                    features.append(mfcc.flatten())
                    labels.append(label_value)
                except Exception as e:
                    print(f"Failed to process {file_path}: {e}")
    return features, labels

# === STEP 3: LOAD DATA FROM LOCAL FOLDERS ===
base_dir = r"./deepfake_detection_dataset_urdu"
bonafide_feat, bonafide_lbl = extract_features_recursive(os.path.join(base_dir, "Bonafide"), 0)
tacotron_feat, tacotron_lbl = extract_features_recursive(os.path.join(base_dir, "Spoofed_Tacotron"), 1)
vits_feat, vits_lbl = extract_features_recursive(os.path.join(base_dir, "Spoofed_TTS"), 1)

X = np.array(bonafide_feat + tacotron_feat + vits_feat)
y = np.array(bonafide_lbl + tacotron_lbl + vits_lbl)

print("Feature shape:", X.shape)
print("Label distribution:", np.bincount(y))

# === STEP 4: SPLIT AND SCALE ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# === STEP 5: CLASSICAL MODELS ===
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train_scaled, y_train)
y_pred_lr = lr.predict(X_test_scaled)
y_prob_lr = lr.predict_proba(X_test_scaled)[:, 1]

svm = SVC(kernel='linear', probability=True)
svm.fit(X_train_scaled, y_train)
y_pred_svm = svm.predict(X_test_scaled)
y_prob_svm = svm.predict_proba(X_test_scaled)[:, 1]

perc = Perceptron(max_iter=1000)
perc.fit(X_train_scaled, y_train)
y_pred_perc = perc.predict(X_test_scaled)

# === STEP 6: DNN ===
dnn = Sequential([
    Dense(256, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.3),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])
dnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
dnn.fit(X_train_scaled, y_train, epochs=15, batch_size=32, validation_split=0.1, verbose=1)
y_prob_dnn = dnn.predict(X_test_scaled).flatten()
y_pred_dnn = (y_prob_dnn > 0.5).astype(int)

def save_model(model, filename):
    try:
        joblib.dump(model, filename)
        print(f"Model saved to {filename}")
    except Exception as e:
        print(f"Failed to save model {filename}: {e}")
        
output_dir = "binary-models"
# Save classical models
# Ensure the directory exists
os.makedirs(output_dir, exist_ok=True)
# Save classical models to the directory
os.makedirs(output_dir, exist_ok=True)
save_model(lr, os.path.join(output_dir, "whis_binary_logistic_regression_model.pkl"))
save_model(svm, os.path.join(output_dir, "whis_binary_svm_model.pkl"))
save_model(perc, os.path.join(output_dir, "whis_binary_perceptron_model.pkl"))

# Save DNN model
try:
    dnn.save(os.path.join(output_dir, "whis_dnn_binary_model.h5"))
    print("DNN model saved to ${output_dir}/dnn_binary_model.h5")
except Exception as e:
    print(f"Failed to save DNN model: {e}")

# === STEP 7: EVALUATION ===
def print_results(name, y_true, y_pred, y_prob=None):
    print(f"\n=== {name} ===")
    print(classification_report(y_true, y_pred))
    if y_prob is not None:
        print("AUC-ROC:", roc_auc_score(y_true, y_prob))

print_results("Logistic Regression", y_test, y_pred_lr, y_prob_lr)
print_results("SVM", y_test, y_pred_svm, y_prob_svm)
print_results("Perceptron", y_test, y_pred_perc)
print_results("DNN", y_test, y_pred_dnn, y_prob_dnn)
