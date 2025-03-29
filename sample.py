import os
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

# Set dataset paths
base_path = "C:/Users/sri sitarama swami/PycharmProjects/Sleep_disorder/dataset"
ecg_folder = os.path.join(base_path, "ECG")

# Load labels
df_ecg = pd.read_csv(os.path.join(base_path, "labels.csv"))

# Ensure image_name column is string type and remove any extensions
df_ecg["image_name"] = df_ecg["image_name"].astype(str).str.replace(".png", "", regex=False)

# Function to load only ECG images
def load_ecg_images(df_ecg, ecg_folder, image_size=(224, 224)):
    images, labels = [], []
    missing_ecg = 0

    for _, row in df_ecg.iterrows():
        img_name = row["image_name"]
        label = row["sleep_disorder"]

        ecg_path = os.path.join(ecg_folder, img_name + ".png")

        # Load ECG image
        ecg_img = cv2.imread(ecg_path) if os.path.exists(ecg_path) else None

        if ecg_img is None:
            missing_ecg += 1
            continue  # Skip missing images instead of replacing

        # Resize and normalize image
        ecg_img = cv2.resize(ecg_img, image_size) / 255.0

        images.append(ecg_img)
        labels.append(label)

    print(f"✅ Total ECG images loaded: {len(images)}")
    print(f"⚠️ Missing ECG images skipped: {missing_ecg}")

    return np.array(images), np.array(labels)

# Load dataset
X, y = load_ecg_images(df_ecg, ecg_folder)

# Convert labels to categorical
y = to_categorical(y)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Print dataset details
print(f"✅ Training set: {X_train.shape}, Labels: {y_train.shape}")
print(f"✅ Test set: {X_test.shape}, Labels: {y_test.shape}")

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Model Architecture
model = Sequential([
    Conv2D(32, (3, 3), activation="relu", input_shape=(224, 224, 3)),  # Only ECG (not 6 channels)
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation="relu"),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(128, (3, 3), activation="relu"),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(256, activation="relu"),
    Dropout(0.5),
    Dense(3, activation="softmax")  # Adjust classes if needed
])

# Compile Model
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Model Summary
model.summary()

# Train the model
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=20, batch_size=16
)

# Save Model
model.save("C:/Users/sri sitarama swami/PycharmProjects/Sleep_disorder/model/sleep_model_ecg.h5")
print("✅ Model training completed and saved successfully!")

# Evaluate on Test Data
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"✅ Test Accuracy: {test_acc * 100:.2f}%")

import numpy as np

# Predict on test set
predictions = model.predict(X_test)
predicted_classes = np.argmax(predictions, axis=1)
actual_classes = np.argmax(y_test, axis=1)

# Show Sample Prediction
for i in range(5):
    print(f"Actual: {actual_classes[i]}, Predicted: {predicted_classes[i]}")
