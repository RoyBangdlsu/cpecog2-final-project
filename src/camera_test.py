import cv2
from tensorflow.keras.models import load_model
import numpy as np

# Load model
model = load_model('models/emotion_detector.keras')

# Emotion labels
labels = ['Angry','Happy', 'Neutral', 'Sad']

# Start video capture
cap = cv2.VideoCapture(0)

emotion = "No face detected"

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml").detectMultiScale(gray)

    for (x, y, w, h) in faces:
        roi = frame[y:y + h, x:x + w]
        img = cv2.resize(roi, (48, 48))
        img = np.expand_dims(img, axis=0) / 255.0

        prediction = model.predict(img)
        emotion = labels[np.argmax(prediction)]

        # Draw rectangle and emotion label on frame
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (0, 255, 0), 2, cv2.LINE_AA)

    # Show live feed with emotion overlay
    cv2.imshow('Emotion Detector', frame)

    # Press 'q' to exit the window
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
