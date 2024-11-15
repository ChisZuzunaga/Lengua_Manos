import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
import mediapipe as mp
import os
import time

global gesture_history, letter_mode, last_gesture, timer, timer_started
# Cargar las clases de las subcarpetas en 'data'
data_dir = "data"
class_names = sorted(os.listdir(data_dir))

# Cargar el modelo entrenado
model = tf.keras.models.load_model("src/sign_language_model.h5")

# Inicializa MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

# Variables globales para el historial y el modo de letras
gesture_history = ""
last_gesture = ""
letter_mode = False
timer = 0
timer_started = False

def recognize_gesture(frame):
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    img_resized = cv2.resize(frame, (128, 128))
    img_array = img_to_array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Realizar la predicción
    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]

    return predicted_class, result.multi_hand_landmarks  # Devolver también las marcas de la mano

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Dimensiones del cuadro guía
        frame_height, frame_width = frame.shape[:2]
        square_size = 300  # Tamaño del cuadrado guía
        x1, y1 = (frame_width - square_size) // 2, (frame_height - square_size) // 2
        x2, y2 = x1 + square_size, y1 + square_size

        # Reconocer el gesto
        predicted_class, hand_landmarks = recognize_gesture(frame)

        # Verificar si la mano está dentro del cuadro
        hand_in_box = False
        if hand_landmarks:
            for hand in hand_landmarks:
                for lm in hand.landmark:
                    x, y = int(lm.x * frame_width), int(lm.y * frame_height)
                    if x1 <= x <= x2 and y1 <= y <= y2:
                        hand_in_box = True
                        break  # Detener la verificación si se encuentra una mano dentro del cuadro

        # Actualizar el historial de gestos solo si la mano está dentro del cuadro
          # Declarar las variables globales aquí
        if hand_in_box:
            last_gesture = predicted_class  # Actualizar el último gesto reconocido

            if letter_mode:
                if not timer_started:
                    timer = 1  # Reiniciar el temporizador a 1 segundo
                    timer_started = True
                else:
                    timer -= 1 / 30  # Disminuir el temporizador basado en el frame rate
                    if timer <= 0:
                        gesture_history += last_gesture  # Agregar el gesto al historial
                        timer_started = False  # Reiniciar el temporizador

        # Mostrar el último gesto si no hay mano en el cuadro
        if not hand_in_box and last_gesture:
            gesture_history = last_gesture  # Mantener el último gesto reconocido

        # Definir posición para el texto centrado en la parte inferior
        text_position = (frame.shape[1] // 2 - 100, frame.shape[0] - 30)

        # Dibujar contorno blanco y luego texto negro
        cv2.putText(frame, last_gesture if hand_in_box else "", text_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(frame, last_gesture if hand_in_box else "", text_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)

        # Mostrar el historial de gestos en la esquina superior izquierda
        cv2.putText(frame, gesture_history, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

        # Mostrar temporizador si está en modo de letras
        if letter_mode and timer_started:
            timer_text = f"Preparar gesto: {int(timer)}"
            cv2.putText(frame, timer_text, (frame_width // 2 - 100, frame_height // 2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        # Dibuja el cuadro guía en el centro
        color = (0, 255, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Mostrar el resultado en pantalla
        cv2.imshow("Reconocimiento de gestos", frame)

        # Controles para activar el modo de letras
        key = cv2.waitKey(1) & 0xFF
        if key == ord('m'):  # Cambiar modo de letras al presionar 'm'
            letter_mode = not letter_mode
            if letter_mode:
                print("Modo letras activado.")
            else:
                print("Modo letras desactivado. Palabra completa:", gesture_history)
                gesture_history = ""  # Reiniciar el historial después de formar la palabra
        elif key == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()