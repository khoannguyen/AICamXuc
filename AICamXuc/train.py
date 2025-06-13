import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
import os
import data
import model
import matplotlib.pyplot as plt

# Lấy đường dẫn và nhãn
file_paths, labels = data.get_file_paths()

# Lưu features cho toàn bộ dataset train/val/test thông qua hàm save_feature

def save_split_features(paths, features_list, split_name, augment_flags=None):
    """
    Lưu features vào thư mục data/features/<split_name>/<emotion>/
    - paths: list đường dẫn file gốc
    - features_list: list hoặc array các feature tương ứng
    - augment_flags: list cùng độ dài paths, chỉ định chỉ số augment hoặc None
    """
    for idx, path in enumerate(paths):
        feats = features_list[idx]
        aug_idx = augment_flags[idx] if augment_flags is not None else None
        data.save_feature(path, feats, augment_idx=aug_idx, split=split_name)

# Mã hóa nhãn
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)
y = tf.keras.utils.to_categorical(labels_encoded)

# Chia dữ liệu thành train_val và test
train_val_paths, test_paths, train_val_y, test_y = train_test_split(
    file_paths, y, test_size=0.2, random_state=42
)

# Trích xuất và lưu features cho tập test
X_test = []
for path in test_paths:
    feats = data.extract_features(path, augment=False)
    X_test.append(feats)
# Lưu features test
save_split_features(test_paths, X_test, 'test')

# Lưu dữ liệu test cho evaluate.py
np.save('data/test_paths.npy', test_paths)
np.save('data/test_y.npy', test_y)

# Chia train_val thành train và val
train_paths, val_paths, train_y, val_y = train_test_split(
    train_val_paths, train_val_y, test_size=0.2, random_state=42
)

# Trích xuất và lưu features cho tập train với augmentation
X_train = []
augment_flags = []
for path in train_paths:
    for aug_idx in range(10):  # Tạo 10 phiên bản augmented
        features = data.extract_features(path, augment=True)
        X_train.append(features)
        augment_flags.append(aug_idx)
# Lưu features train
save_split_features(np.repeat(train_paths, 10), X_train, 'train', augment_flags)

# Tạo y cho train augmented
train_y_aug = np.repeat(train_y, 10, axis=0)

# Trích xuất và lưu features cho tập val
X_val = []
for path in val_paths:
    feats = data.extract_features(path, augment=False)
    X_val.append(feats)
# Lưu features val
save_split_features(val_paths, X_val, 'val')

# Chuyển sang mảng numpy
X_train = np.array(X_train)
X_val = np.array(X_val)
X_test = np.array(X_test)

# Tính toán trọng số lớp
train_labels = np.argmax(train_y, axis=1)
class_weights = compute_class_weight(
    'balanced', classes=np.unique(train_labels), y=train_labels
)
class_weights_dict = dict(enumerate(class_weights))

# Xây dựng mô hình
input_shape = X_train.shape[1:]  # (time_steps, n_features)
num_classes = y.shape[1]
cnn_model = model.build_model(input_shape, num_classes)

# Biên dịch mô hình
cnn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Early stopping
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=5, restore_best_weights=True
)

# Huấn luyện mô hình
history = cnn_model.fit(
    X_train, train_y_aug,
    epochs=50,
    batch_size=32,
    validation_data=(X_val, val_y),
    callbacks=[early_stopping],
    class_weight=class_weights_dict
)

# Lưu mô hình
cnn_model.save('model/emotion_recognition_cnn.keras')
print("Mô hình đã được lưu vào 'emotion_recognition_cnn.keras'")

# Lưu label encoder
data.save_label_encoder(label_encoder)

# Đánh giá độ chính xác trên tập huấn luyện và tập kiểm tra
train_loss, train_accuracy = cnn_model.evaluate(X_train, train_y_aug)
test_loss, test_accuracy = cnn_model.evaluate(X_test, test_y)

print(f"Độ chính xác trên tập dữ liệu huấn luyện: {train_accuracy * 100:.2f}%")
print(f"Độ chính xác trên tập dữ liệu kiểm tra: {test_accuracy * 100:.2f}%")

# Vẽ biểu đồ lịch sử huấn luyện
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.savefig('evaluate/accuracy_plot.png')
plt.close()

plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.savefig('evaluate/loss_plot.png')
plt.close()