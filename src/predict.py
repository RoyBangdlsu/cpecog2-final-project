from tensorflow.keras.models import load_model
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pickle

# ✅ Load model
model = load_model('models/emotion_detector.keras')

# ✅ Emotion labels
labels = ['Angry', 'Disgusted', 'Fearful', 'Happy', 'Neutral', 'Sad', 'Surprised']

# ✅ Function to Predict Emotion
def predict_emotion(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (48, 48))
    img = np.expand_dims(img, axis=0) / 255.0

    prediction = model.predict(img)[0]  # Get the softmax output directly
    emotion = labels[np.argmax(prediction)]

    # ✅ Plot Prediction Result
    plt.figure(figsize=(8, 5))
    plt.bar(labels, prediction, color='skyblue')
    plt.title(f"Predicted Emotion: {emotion}")
    plt.xlabel('Emotions')
    plt.ylabel('Probability')
    plt.ylim(0, 1)
    plt.show()

    return emotion

# ✅ Function to Plot Training History
def plot_training_history():
    with open('models/history.pkl', 'rb') as f:
        history = pickle.load(f)

    plt.figure(figsize=(12, 5))

    # ✅ Accuracy Plot
    plt.subplot(1, 2, 1)
    plt.plot(history['accuracy'], label='Training Accuracy', marker='o')
    plt.plot(history['val_accuracy'], label='Validation Accuracy', marker='o')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # ✅ Loss Plot
    plt.subplot(1, 2, 2)
    plt.plot(history['loss'], label='Training Loss', marker='o')
    plt.plot(history['val_loss'], label='Validation Loss', marker='o')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()

# ✅ Test Example
image_path = 'src/test_happy.jpg'
emotion = predict_emotion(image_path)
print(f"Predicted Emotion: {emotion}")

# ✅ Plot Training History
plot_training_history()
