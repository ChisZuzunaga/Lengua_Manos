import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
import mediapipe as mp
import os  # Para listar las carpetas dinámicamente

# Cargar las clases de las subcarpetas en 'data'
data_dir = "data"
class_names = sorted(os.listdir(data_dir))  # Lista todas las carpetas dentro de 'data'

# Cargar el modelo entrenado
model = tf.keras.models.load_model("src/sign_language_model.h5")

# Inicializa MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

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
    
    return predicted_class

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        predicted_class = recognize_gesture(frame)
        cv2.putText(frame, predicted_class, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.imshow("Reconocimiento de gestos", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
