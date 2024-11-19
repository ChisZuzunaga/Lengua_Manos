import tkinter as tk
from tkinter import ttk
import os
import threading

# Obtener la ruta del directorio actual
current_dir = os.path.dirname(os.path.abspath(__file__))

# Crear la ventana principal
root = tk.Tk()
root.title("Entrenamiento de Modelo de Lenguaje de Señas")
root.geometry("400x400")
root.configure(bg="#1E88E5")  # Azul claro de fondo

# Función para centrar una ventana
def center_window(window, width, height):
    screen_width = window.winfo_screenwidth()
    screen_height = window.winfo_screenheight()
    x = (screen_width - width) // 2
    y = (screen_height - height) // 2
    window.geometry(f"{width}x{height}+{x}+{y}")

# Centrar la ventana principal
center_window(root, 400, 400)

# Variables globales para la configuración
num_epochs = tk.IntVar(value=10)  # Valor por defecto
camera_index = tk.IntVar(value=0)  # Valor por defecto

# Estilos generales
btn_style = {
    "bg": "#FF7043",  # Naranja
    "fg": "white",
    "activebackground": "#FF5722",  # Naranja oscuro
    "activeforeground": "white",
    "font": ("Helvetica", 12, "bold"),
    "relief": "raised",
    "bd": 2,
    "width": 20
}

label_style = {
    "bg": "#1E88E5",  # Azul claro
    "fg": "white",
    "font": ("Helvetica", 12, "bold")
}

entry_style = {
    "bg": "#E3F2FD",  # Celeste claro
    "fg": "#0D47A1",  # Azul oscuro
    "font": ("Helvetica", 12),
    "relief": "flat",
    "bd": 2
}

# Animación para botones
def on_enter(e):
    e.widget["bg"] = "#FFAB91"

def on_leave(e):
    e.widget["bg"] = "#FF7043"

# Crear un contenedor para centrar los botones principales
frame_buttons = tk.Frame(root, bg="#1E88E5")
frame_buttons.pack(expand=True, fill="both")  # Centrar el contenedor automáticamente

# Funciones de la interfaz
def open_capture_window():
    capture_window = tk.Toplevel(root)
    capture_window.title("Capturar Datos")
    center_window(capture_window, 400, 300)
    capture_window.configure(bg="#1E88E5")

    tk.Label(capture_window, text="Ingrese el gesto:", **label_style).pack(pady=10)
    gesture_entry = tk.Entry(capture_window, **entry_style)
    gesture_entry.pack(pady=10)

    def start_capture():
        gesture = gesture_entry.get()
        if gesture:
            capture_window.destroy()

    start_button = tk.Button(capture_window, text="Iniciar Captura", command=start_capture, **btn_style)
    start_button.pack(pady=10)

    # Animaciones para el botón
    start_button.bind("<Enter>", on_enter)
    start_button.bind("<Leave>", on_leave)

def run_train_model():
    training_window = tk.Toplevel(root)
    training_window.title("Entrenando Modelo")
    center_window(training_window, 350, 120)
    training_window.configure(bg="#1E88E5")

    progress_bar = ttk.Progressbar(training_window, orient="horizontal", length=250, mode="determinate")
    progress_bar.pack(pady=20)

    def train_model_thread():
        training_window.destroy()

    threading.Thread(target=train_model_thread).start()

def run_gesture_recognition():
    recognition_window = tk.Toplevel(root)
    recognition_window.title("Reconocimiento de Gestos")
    center_window(recognition_window, 400, 300)
    recognition_window.configure(bg="#1E88E5")
    tk.Label(recognition_window, text="Reconocimiento de Gestos en Progreso...", **label_style).pack(pady=20)

def configure_settings():
    settings_window = tk.Toplevel(root)
    settings_window.title("Configuración")
    center_window(settings_window, 300, 200)
    settings_window.configure(bg="#1E88E5")

    frame = tk.Frame(settings_window, bg="#1E88E5")
    frame.pack(expand=True)  # Expand para centrar vertical y horizontalmente

    tk.Label(frame, text="Número de Épocas:", **label_style).pack(pady=5)
    epochs_entry = tk.Entry(frame, textvariable=num_epochs, **entry_style)
    epochs_entry.pack(pady=5)

    tk.Label(frame, text="Seleccionar Cámara:", **label_style).pack(pady=5)
    cameras = [i for i in range(5)]
    camera_dropdown = tk.OptionMenu(frame, camera_index, *cameras)
    camera_dropdown.config(font=("Helvetica", 12), bg="#E3F2FD", fg="#0D47A1")  # Personalizar menú
    camera_dropdown.pack(pady=5)

    save_button = tk.Button(frame, text="Guardar", command=settings_window.destroy, **btn_style)
    save_button.pack(pady=10)

    save_button.bind("<Enter>", on_enter)
    save_button.bind("<Leave>", on_leave)

# Botones en el contenedor centrado
btn_capture_data = tk.Button(frame_buttons, text="Capturar Datos", command=open_capture_window, **btn_style)
btn_capture_data.pack(pady=10)
btn_capture_data.bind("<Enter>", on_enter)
btn_capture_data.bind("<Leave>", on_leave)

btn_train_model = tk.Button(frame_buttons, text="Entrenar Modelo", command=run_train_model, **btn_style)
btn_train_model.pack(pady=10)
btn_train_model.bind("<Enter>", on_enter)
btn_train_model.bind("<Leave>", on_leave)

btn_recognize_gesture = tk.Button(frame_buttons, text="Reconocer Gestos", command=run_gesture_recognition, **btn_style)
btn_recognize_gesture.pack(pady=10)
btn_recognize_gesture.bind("<Enter>", on_enter)
btn_recognize_gesture.bind("<Leave>", on_leave)

btn_settings = tk.Button(frame_buttons, text="Configuración", command=configure_settings, **btn_style)
btn_settings.pack(pady=10)
btn_settings.bind("<Enter>", on_enter)
btn_settings.bind("<Leave>", on_leave)

# Iniciar la ventana principal
root.mainloop()
