import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import os
import librosa
import data  # Giả định có module data.py với extract_features và load_label_encoder

# Tạo thư mục evaluate nếu chưa có
os.makedirs("evaluate", exist_ok=True)

# Tải mô hình đã huấn luyện và label encoder
model = tf.keras.models.load_model("model/emotion_recognition_cnn.keras")
label_encoder = data.load_label_encoder(filename="data/label_encoder.joblib")

# Tải đường dẫn và nhãn test đã lưu
test_paths = np.load('data/test_paths.npy', allow_pickle=True)
y_test = np.load('data/test_y.npy')

# Trích xuất đặc trưng từ test paths
X_test = [data.extract_features(path, augment=False) for path in test_paths]
X_test = np.array(X_test)

# Chuyển nhãn sang định dạng số
y_test_labels = np.argmax(y_test, axis=1)
class_names = label_encoder.classes_

# Dự đoán trên tập test
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)

# Tính toán các chỉ số hiệu suất tổng thể
accuracy = accuracy_score(y_test_labels, y_pred)
precision = precision_score(y_test_labels, y_pred, average='macro')
recall = recall_score(y_test_labels, y_pred, average='macro')
f1 = f1_score(y_test_labels, y_pred, average='macro')

# Ghi kết quả tổng thể vào file
with open("evaluate/result.txt", "w", encoding="utf-8") as f:
    f.write(f"Tổng số mẫu thử nghiệm: {len(y_test_labels)}\n")
    f.write("Kết quả đánh giá tổng thể:\n")
    f.write(f"Accuracy: {accuracy * 100:.2f}%\n")
    f.write(f"Precision: {precision:.4f}\n")
    f.write(f"Recall: {recall:.4f}\n")
    f.write(f"F1-Score: {f1:.4f}\n\n")

    f.write("Kết quả đánh giá cho từng lớp cảm xúc:\n")
    report = classification_report(y_test_labels, y_pred, target_names=class_names)
    f.write(report)

# Vẽ và lưu ma trận nhầm lẫn
cm = confusion_matrix(y_test_labels, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.title('Ma trận nhầm lẫn')
plt.xlabel('Dự đoán')
plt.ylabel('Thực tế')
plt.tight_layout()
plt.savefig("evaluate/confusion_matrix.png")
plt.close()

# --- Bổ sung: Phân phối nhãn (Bar & Pie) ---
counts = np.bincount(y_test_labels)
plt.figure(figsize=(8, 5))
plt.bar(class_names, counts)
plt.title('Phân phối nhãn (Tập kiểm tra)')
plt.xlabel('Lớp cảm xúc')
plt.ylabel('Số lượng mẫu')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('evaluate/label_distribution_bar.png')
plt.close()

plt.figure(figsize=(6, 6))
plt.pie(counts, labels=class_names, autopct='%1.1f%%', startangle=90)
plt.title('Tỷ lệ phân bổ cảm xúc')
plt.tight_layout()
plt.savefig('evaluate/label_distribution_pie.png')
plt.close()

# --- Bổ sung: So sánh tín hiệu âm thanh (Waveform) ---
plt.figure(figsize=(12, 8))
for idx, cls in enumerate(class_names):
    sample_path = test_paths[y_test_labels == idx][0]
    signal, sr = librosa.load(sample_path, sr=None)
    time = np.linspace(0, len(signal) / sr, num=len(signal))
    plt.subplot((len(class_names)+1)//2, 2, idx+1)
    plt.plot(time, signal)
    plt.title(f'Sóng âm - Cảm xúc: {cls}')
    plt.ylabel('Biên độ')
    if idx >= len(class_names)-2:
        plt.xlabel('Thời gian (s)')
plt.tight_layout()
plt.savefig('evaluate/waveform_comparison.png')
plt.close()

# --- Bổ sung: T-SNE phân bố dữ liệu trong không gian 2D ---
from sklearn.manifold import TSNE
features_flat = np.array(X_test).reshape(X_test.shape[0], -1)
tsne = TSNE(n_components=2, random_state=42)
features_tsne = tsne.fit_transform(features_flat)
plt.figure(figsize=(8, 6))
for idx, cls in enumerate(class_names):
    mask = (y_test_labels == idx)
    plt.scatter(features_tsne[mask, 0], features_tsne[mask, 1], label=cls, alpha=0.7)
plt.legend()
plt.title('t-SNE: Phân bố cảm xúc trong không gian đặc trưng 2D')
plt.xlabel('t-SNE feature 1')
plt.ylabel('t-SNE feature 2')
plt.tight_layout()
plt.savefig('evaluate/tsne_scatter.png')
plt.close()

# Thông báo hoàn thành
print("Hoàn thành đánh giá kết quả lưu trong 'evaluate/'")
