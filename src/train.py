import sys
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ✅ Fix Unicode error on Windows
sys.stdout.reconfigure(encoding='utf-8')

# Path to dataset
DATASET_PATH = r'C:\Users\bsg99\Downloads\cpecog2 project\archive'

# ✅ Data preprocessing and augmentation
data_generator = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,  # Increased rotation for better generalization
    width_shift_range=0.1, 
    height_shift_range=0.1,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2  # Use 20% of training set for validation
)

train_data = data_generator.flow_from_directory(
    DATASET_PATH + '\\train\\',
    target_size=(48, 48),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

val_data = data_generator.flow_from_directory(
    DATASET_PATH + '\\test\\',
    target_size=(48, 48),
    batch_size=32,
    class_mode='categorical',
)

# ✅ Build deeper CNN model
model = models.Sequential([
    # First convolutional block
    layers.Conv2D(64, (3, 3), activation='relu', input_shape=(48, 48, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.2),

    # Second convolutional block
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.2),

    # Third convolutional block
    layers.Conv2D(256, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.3),

    # Fourth convolutional block (for deeper feature extraction)
    layers.Conv2D(512, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.3),

    # Fully connected layers
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.4),  # Stronger dropout to prevent overfitting
    layers.Dense(7, activation='softmax')  # 7 output classes
])

# ✅ Compile model with lower learning rate
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),  
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# ✅ Train the model for more epochs
history = model.fit(train_data, epochs=50, validation_data=val_data)

# ✅ Save model in the new .keras format
model.save('models/emotion_detector.keras')

print("✅ Training complete. Model saved to 'models/emotion_detector.keras'")


import pickle

with open('models/history.pkl', 'wb') as f:
    pickle.dump(history.history, f)
