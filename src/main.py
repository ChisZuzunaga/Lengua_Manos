import tkinter as tk
from tkinter import ttk
import os
import threading
import subprocess  # Asegúrate de importar subprocess
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models

# Obtener la ruta del directorio actual
current_dir = os.path.dirname(os.path.abspath(__file__))

# Crear la ventana principal
root = tk.Tk()
root.title("Entrenamiento de Modelo de Lenguaje de Señas")
root.geometry("400x300")

# Variables globales para la configuración
num_epochs = tk.IntVar(value=10)  # Valor por defecto
camera_index = tk.IntVar(value=0)  # Valor por defecto

# Funciones de entrenamiento
DATA_DIR = "data"
IMG_SIZE = (128, 128)
BATCH_SIZE = 32

class ProgressCallback(tf.keras.callbacks.Callback):
    def __init__(self, update_progress):
        super().__init__()
        self.update_progress = update_progress

    def on_epoch_end(self, epoch, logs=None):
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
    
    progress_callback = ProgressCallback(update_progress)
    
    model.fit(
        train_data,
        validation_data=val_data,
        epochs=num_epochs,
        callbacks=[progress_callback]
    )
    
    model.save("src/sign_language_model.h5")

# Funciones de la interfaz
def run_capture_data(gesture):
    btn_capture_data.config(state=tk.DISABLED)

    def capture_data_thread():
        subprocess.run(["python", os.path.join(current_dir, "capture_data.py"), str(camera_index.get()), gesture])
        btn_capture_data.config(state=tk.NORMAL)

    threading.Thread(target=capture_data_thread).start()

def open_capture_window():
    capture_window = tk.Toplevel(root)
    capture_window.title("Capturar Datos")

    tk.Label(capture_window, text="Ingrese el gesto:").pack(pady=10)
    gesture_entry = tk.Entry(capture_window)
    gesture_entry.pack(pady=10)

    def start_capture():
        gesture = gesture_entry.get()
        if gesture:
            run_capture_data(gesture)
            capture_window.destroy()

    start_button = tk.Button(capture_window, text="Iniciar Captura", command=start_capture)
    start_button.pack(pady=10)

def run_train_model():
    training_window = tk.Toplevel(root)
    training_window.title("Entrenando Modelo")
    training_window.geometry("300x100")

    progress_bar = ttk.Progressbar(training_window, orient="horizontal", length=250, mode="determinate")
    progress_bar.pack(pady=20)

    def train_model_thread():
        def update_progress(epoch):
            progress_bar['value'] = (epoch / num_epochs.get()) * 100
            training_window.update_idletasks()

        train_model(num_epochs.get(), update_progress)  # Llamar a train_model directamente

        training_window.destroy()  # Cierra la ventana al finalizar el entrenamiento

    threading.Thread(target=train_model_thread).start()

def run_gesture_recognition():
    subprocess.run(["python", os.path.join(current_dir, "gesture_recognition.py"), str(camera_index.get())])

def configure_settings():
    settings_window = tk.Toplevel(root)
    settings_window.title("Configuración")

    # Configuración de número de épocas
    tk.Label(settings_window, text="Número de Épocas:").pack(pady=5)
    epochs_entry = tk.Entry(settings_window, textvariable=num_epochs)
    epochs_entry.pack(pady=5)

    # Configuración de cámara
    tk.Label(settings_window, text="Seleccionar Cámara:").pack(pady=5)
    cameras = [i for i in range(5)]  # Cambia el rango según el número de cámaras que esperas
    camera_dropdown = tk.OptionMenu(settings_window, camera_index, *cameras)
    camera_dropdown.pack(pady=5)

    # Botón para guardar la configuración
    save_button = tk.Button(settings_window, text="Guardar", command=settings_window.destroy)
    save_button.pack(pady=10)

def recognize_gesture():
    run_gesture_recognition()

def open_settings():
    configure_settings()

# Botones y elementos de la interfaz
btn_capture_data = tk.Button(root, text="Capturar Datos", command=open_capture_window)
btn_capture_data.pack(pady=20)

btn_train_model = tk.Button(root, text="Entrenar Modelo", command=run_train_model)
btn_train_model.pack(pady=20)

btn_recognize_gesture = tk.Button(root, text="Reconocer Gestos", command=recognize_gesture)
btn_recognize_gesture.pack(pady=20)

btn_settings = tk.Button(root, text="Configuración", command=open_settings)
btn_settings.pack(pady=20)

root.mainloop()