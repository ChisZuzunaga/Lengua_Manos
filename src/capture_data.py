import cv2
import os
import mediapipe as mp
import time  # Importa la librería para medir el tiempo

# Inicializa MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

# Carpeta para almacenar las imágenes
DATA_DIR = "data"

def create_gesture_folder(gesture_name):
    folder_path = os.path.join(DATA_DIR, gesture_name)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    return folder_path

def capture_images(gesture_name, capture_time=5):
    cap = cv2.VideoCapture(0)
    folder_path = create_gesture_folder(gesture_name)

    img_count = len(os.listdir(folder_path)) if os.path.exists(folder_path) else 0  # Contar imágenes existentes

    start_time = time.time()  # Captura el tiempo de inicio
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error al capturar el frame.")
            break
        
        # Convierte la imagen a RGB y procesa con MediaPipe
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(img_rgb)
        
        # Dibuja las manos detectadas
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        
        # Muestra la imagen en la ventana
        cv2.imshow("Captura de gestos", frame)
        
        # Guarda cada imagen capturada con un nombre único
        img_path = os.path.join(folder_path, f"{gesture_name}_{img_count}.png")
        cv2.imwrite(img_path, frame)
        print(f"Imagen guardada: {img_path}")  # Mensaje de confirmación
        img_count += 1  # Incrementa el contador después de guardar la imagen
        
        # Finaliza la captura después de un tiempo específico
        if time.time() - start_time >= capture_time:
            print("Tiempo de captura finalizado.")
            break
        
        # Finaliza la captura con la tecla 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    gesture_name = input("Introduce el nombre del gesto o letra a capturar: ")
    capture_images(gesture_name, capture_time=5)  # Captura durante 10 segundos
