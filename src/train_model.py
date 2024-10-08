import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models

DATA_DIR = "data"
IMG_SIZE = (128, 128)
BATCH_SIZE = 32

def load_data():
    datagen = ImageDataGenerator(rescale=1.0/255.0, validation_split=0.2)
    
    train_data = datagen.flow_from_directory(
        DATA_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training'
    )
    
    val_data = datagen.flow_from_directory(
        DATA_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation'
    )
    
    return train_data, val_data

def create_model(num_classes):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def train_model():
    train_data, val_data = load_data()
    num_classes = len(train_data.class_indices)
    
    model = create_model(num_classes)
    
    model.fit(
        train_data,
        validation_data=val_data,
        epochs=12
    )
    
    model.save("src/sign_language_model.h5")

if __name__ == "__main__":
    train_model()
