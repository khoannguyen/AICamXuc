import os
import numpy as np
import librosa
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift
import joblib

# Cài đặt tham số
SAMPLE_RATE = 16000
DURATION = 5
N_MFCC = 40
AUDIO_PATH = "data/train/"
EMOTIONS = ["vui", "buon", "tucgian", "sohai", "ngacnhien", "trungtinh"]

# Pipeline tăng cường dữ liệu
augment_pipeline = Compose([
    AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
    TimeStretch(min_rate=0.9, max_rate=1.1, p=0.5),
    PitchShift(min_semitones=-2, max_semitones=2, p=0.5),
])

def extract_features(file_path, augment=False):
    audio, sr = librosa.load(file_path, sr=SAMPLE_RATE)
    if augment:
        audio = augment_pipeline(audio, sample_rate=sr)

    # Chuẩn hóa độ dài
    target_len = SAMPLE_RATE * DURATION
    if len(audio) > target_len:
        audio = audio[:target_len]
    else:
        audio = np.pad(audio, (0, target_len - len(audio)))

    # Trích xuất đặc trưng MFCC và delta
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC)
    delta_mfcc = librosa.feature.delta(mfcc)
    delta2_mfcc = librosa.feature.delta(mfcc, order=2)

    # Pitch (cao độ)
    pitch = librosa.yin(audio, fmin=50, fmax=300, sr=sr)
    pitch = np.pad(pitch.reshape(-1, 1), ((0, max(0, mfcc.shape[1] - pitch.shape[0])), (0, 0)), mode='edge')[:mfcc.shape[1]]

    # RMS (cường độ)
    rms = librosa.feature.rms(y=audio).T
    rms = np.pad(rms, ((0, max(0, mfcc.shape[1] - rms.shape[0])), (0, 0)), mode='edge')[:mfcc.shape[1]]

    # ZCR (zero crossing rate)
    zcr = librosa.feature.zero_crossing_rate(y=audio).T
    zcr = np.pad(zcr, ((0, max(0, mfcc.shape[1] - zcr.shape[0])), (0, 0)), mode='edge')[:mfcc.shape[1]]

    # Tempo (toàn đoạn)
    tempo, _ = librosa.beat.beat_track(y=audio, sr=sr)
    tempo_arr = np.full((mfcc.shape[1], 1), tempo)

    # Gộp đặc trưng theo trục thời gian
    features = np.hstack([
        mfcc.T,
        delta_mfcc.T,
        delta2_mfcc.T,
        pitch,
        rms,
        zcr,
        tempo_arr
    ])
    return features


def save_feature(file_path, features, augment_idx=None, split=None):
    """
    Lưu mảng features vào thư mục data/features/<split>/<emotion>.
    Nếu augment_idx không None, thêm hậu tố _aug{augment_idx} vào tên file.
    """
    # Xác định emotion từ đường dẫn gốc
    emotion = os.path.basename(os.path.dirname(file_path))
    # Thư mục đích
    base_dir = os.path.join('data', 'features', split, emotion)
    os.makedirs(base_dir, exist_ok=True)
    # Tên file
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    if augment_idx is not None:
        file_name = f"{base_name}_aug{augment_idx}.npy"
    else:
        file_name = f"{base_name}.npy"
    # Đường dẫn đầy đủ
    save_path = os.path.join(base_dir, file_name)
    # Lưu file numpy
    np.save(save_path, features)


def get_file_paths():
    file_paths, labels = [], []
    for emotion in EMOTIONS:
        folder = os.path.join(AUDIO_PATH, emotion)
        for file in os.listdir(folder):
            if file.endswith('.wav'):
                file_paths.append(os.path.join(folder, file))
                labels.append(emotion)
    return file_paths, labels


def save_label_encoder(encoder, filename="data/label_encoder.joblib"):
    joblib.dump(encoder, filename)


def load_label_encoder(filename="data/label_encoder.joblib"):
    return joblib.load(filename)

if __name__ == "__main__":
    file_paths, labels = get_file_paths()
    
    print(f"Tổng số file âm thanh: {len(file_paths)}")
    print("Trích xuất đặc trưng các dữ liệu âm thanh:")

    for i in range(min(1410, len(file_paths))):
        print(f"\nFile: {file_paths[i]}")
        features = extract_features(file_paths[i])
        print(f"  Nhãn: {labels[i]}")
        print(f"  Kích thước đặc trưng: {features.shape}")
        print(f"  Một vài giá trị đầu tiên:\n{features[:2]}")  # hiển thị 2 frame đầu tiên
