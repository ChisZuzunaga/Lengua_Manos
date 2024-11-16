import cv2
import os
import mediapipe as mp
import sys

# Inicializa MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

# Carpeta para almacenar las imágenes
DATA_DIR = "data"

# Función para crear la carpeta del gesto
def create_gesture_folder(gesture_name):
    folder_path = os.path.join(DATA_DIR, gesture_name)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    return folder_path

# Función para obtener el último número de archivo en la carpeta
def get_last_image_number(folder_path, gesture_name):
    image_files = [f for f in os.listdir(folder_path) if f.startswith(gesture_name) and f.endswith(".png")]
    if not image_files:
        return 0
    # Extrae el número de cada archivo y encuentra el máximo
    numbers = [int(f.split("_")[-1].split(".")[0]) for f in image_files]
    return max(numbers) + 1

# Función para capturar imágenes
def capture_images(gesture_name, camera_index, max_images=60):
    cap = cv2.VideoCapture(camera_index)
    folder_path = create_gesture_folder(gesture_name)

    # Inicializar el contador desde el último número existente en la carpeta
    img_count = get_last_image_number(folder_path, gesture_name)
    new_images_captured = 0  # Contador de nuevas imágenes capturadas

    while new_images_captured < max_images:
        ret, frame = cap.read()
        if not ret:
            print("Error al capturar el frame.")
            break

        # Dimensiones del cuadro guía
        frame_height, frame_width = frame.shape[:2]
        square_size = 300  # Tamaño del cuadrado guía
        x1, y1 = (frame_width - square_size) // 2, (frame_height - square_size) // 2
        x2, y2 = x1 + square_size, y1 + square_size

        # Convierte la imagen a RGB y procesa con MediaPipe
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(img_rgb)

        # Verifica si la mano está dentro del cuadro
        hand_in_box = False
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Verifica si las coordenadas de la mano están dentro del cuadro
                for lm in hand_landmarks.landmark:
                    x, y = int(lm.x * frame_width), int(lm.y * frame_height)
                    if x1 <= x <= x2 and y1 <= y <= y2:
                        hand_in_box = True
                    else:
                        hand_in_box = False
                        break  # Detiene la verificación si un punto está fuera del cuadro

        # Dibuja el cuadro guía en el centro
        color = (0, 255, 0) if hand_in_box else (0, 0, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Guarda la imagen si la mano está dentro del cuadro y el contador de nuevas imágenes es menor al límite
        if hand_in_box and new_images_captured < max_images:
            img_path = os.path.join(folder_path, f"{gesture_name}_{img_count}.png")
            cv2.imwrite(img_path, frame)
            print(f"Imagen guardada: {img_path}")
            img_count += 1
            new_images_captured += 1  # Incrementa el contador de nuevas imágenes

        # Muestra la imagen en la ventana
        cv2.imshow("Captura de gestos", frame)

        # Finaliza la captura con la tecla 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Captura completa. Se han guardado {new_images_captured} imágenes.")

if __name__ == "__main__":
    # Obtener el gesto y el índice de la cámara de los argumentos
    if len (sys.argv) < 3:
        print("Uso: python capture_data.py <camera_index> <gesture_name>")
        sys.exit(1)

    camera_index = int(sys.argv[1])
    gesture_name = sys.argv[2]
    capture_images(gesture_name, camera_index, max_images=60)  # Captura un total de 60 imágenes