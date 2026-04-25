import numpy as np
import cv2
import json
import time
import sys
import os
import onnxruntime as ort

MODEL_V3 = "model_v3_fp16.onnx"
MODEL_V2 = "model_v2_fp16.onnx"
THRESHOLD = 0.75

# -----------------------------
# LOAD CLASS NAMES
# -----------------------------
with open("class_names.json", "r") as f:
    class_names = json.load(f)

# -----------------------------
# LOAD MODELS
# -----------------------------
providers = ort.get_available_providers()

session_v3 = ort.InferenceSession(MODEL_V3, providers=providers)
session_v2 = ort.InferenceSession(MODEL_V2, providers=providers)

input_v3 = session_v3.get_inputs()[0].name
output_v3 = session_v3.get_outputs()[0].name
dtype_v3 = session_v3.get_inputs()[0].type

input_v2 = session_v2.get_inputs()[0].name
output_v2 = session_v2.get_outputs()[0].name
dtype_v2 = session_v2.get_inputs()[0].type


# -----------------------------
# PREPROCESS
# -----------------------------
def preprocess(frame, dtype):
    img = cv2.resize(frame, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 127.5 - 1.0
    img = np.expand_dims(img, axis=0)
    return img.astype(np.float16 if "float16" in dtype else np.float32)


def infer(session, input_name, output_name, img):
    return session.run([output_name], {input_name: img})[0][0]


def compute_confidence(probs):
    probs = probs.astype(np.float32)
    probs = probs / np.sum(probs)
    entropy = -np.sum(probs * np.log(probs + 1e-9))
    return float(1 - entropy / np.log(len(probs)))


# -----------------------------
# CAMERA INIT
# -----------------------------
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("[ERROR] Camera not detected")
    exit()

print("✅ Camera started. Press ESC to exit.")

# -----------------------------
# MAIN LOOP
# -----------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    start_time = time.time()

    # ---- V3 INFERENCE ----
    img_v3 = preprocess(frame, dtype_v3)
    preds_v3 = infer(session_v3, input_v3, output_v3, img_v3)

    class_v3 = int(np.argmax(preds_v3))
    conf_v3 = compute_confidence(preds_v3)

    # ---- MODEL DECISION ----
    if conf_v3 >= THRESHOLD:
        final_class = class_v3
        confidence = conf_v3
        used_model = "MobileNetV3 (Fast)"
    else:
        img_v2 = preprocess(frame, dtype_v2)
        preds_v2 = infer(session_v2, input_v2, output_v2, img_v2)

        final_class = int(np.argmax(preds_v2))
        confidence = compute_confidence(preds_v2)
        used_model = "MobileNetV2 (Accurate)"

    prediction = class_names[final_class]

    # ---- METRICS ----
    latency = (time.time() - start_time) * 1000
    fps = 1 / (time.time() - start_time + 1e-6)

    # -----------------------------
    # DISPLAY OUTPUT
    # -----------------------------
    cv2.putText(frame,
                f"Prediction: {prediction}",
                (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2)

    cv2.putText(frame,
                f"Confidence: {confidence*100:.2f}%",
                (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2)

    cv2.putText(frame,
                f"Model: {used_model}",
                (10, 130),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 0, 0),
                2)

    cv2.putText(frame,
                f"Latency: {latency:.2f} ms",
                (10, 170),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2)

    cv2.putText(frame,
                f"FPS: {fps:.2f}",
                (10, 210),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2)

    cv2.imshow("RSWaD Edge AI - Real-Time Inference", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
