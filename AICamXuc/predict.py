import numpy as np
import tensorflow as tf
from data import extract_features, load_label_encoder

# Ngưỡng cho từng mức độ
THRESHOLDS = {
    "Cảm xúc rất rõ ràng": 0.6,
    "Cảm xúc trung bình": 0.3,
    "Cảm xúc không rõ ràng": 0.1,
}

def predict_emotion(wav_path,
                    model_path="model/emotion_recognition_cnn.keras",
                    le_path="data/label_encoder.joblib"):
    # 1. Load model & encoder
    model = tf.keras.models.load_model(model_path)
    le = load_label_encoder(le_path)

    # 2. Trích feature & predict
    mfcc = extract_features(wav_path, augment=False)
    probs = model.predict(np.expand_dims(mfcc, 0), verbose=0)[0]

    # 3. Phân cấp và gom nhãn
    result = {label: float(p) for label, p in zip(le.classes_, probs)}
    buckets = {level: [] for level in THRESHOLDS}

    for label, p in result.items():
        for level, thr in THRESHOLDS.items():
            if p >= thr:
                buckets[level].append((label, p))
                break  # chỉ xếp vào cấp cao nhất phù hợp

    return buckets

def print_results(buckets):
    print("Kết quả nhận diện:")
    for level, items in buckets.items():
        # Nếu trong bucket không có phần tử thì bỏ qua
        if not items:
            continue

        for label, p in items:
            print(f"{level}: {int(p*100)}% {label}")

if __name__ == "__main__":
    wav_file = "data/test/vui/recordv001.wav"
    buckets = predict_emotion(wav_file)
    print_results(buckets)
