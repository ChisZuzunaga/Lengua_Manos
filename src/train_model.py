import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
import sys

DATA_DIR = "data"
IMG_SIZE = (128, 128)
BATCH_SIZE = 32

class ProgressCallback(tf.keras.callbacks.Callback):
    def __init__(self, update_progress):
        super().__init__()
        self.update_progress = update_progress

    def on_epoch_end(self, epoch, logs=None):
        # Llama a la función de actualización de progreso
        self.update_progress(epoch + 1)

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

def train_model(num_epochs, update_progress):
    train_data, val_data = load_data()
    num_classes = len(train_data.class_indices)
    
    model = create_model(num_classes)
    
    # Crear el callback para actualizar el progreso
    progress_callback = ProgressCallback(update_progress)
    
    model.fit(
        train_data,
        validation_data=val_data,
        epochs=num_epochs,
        callbacks=[progress_callback]  # Añadir el callback aquí
    )
    
    model.save("src/sign_language_model.h5")

if __name__ == "__main__":
    num_epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    # Aquí se debe pasar una función de actualización de progreso
    train_model(num_epochs, lambda epoch: None)  # Placeholder, se reemplazará en main.py