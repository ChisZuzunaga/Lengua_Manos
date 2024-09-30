import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
import pandas as pd

# Cargar el modelo entrenado
model = tf.keras.models.load_model('lengua_manos/models/modelo_gestos.h5')

# Cargar etiquetas para las predicciones
data = pd.read_csv('lengua_manos/data/gestures_data.csv')
labels = data['label'].unique()

# Inicializar MediaPipe y OpenCV
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)

with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Error al acceder a la cámara.")
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Extraer los landmarks
                landmarks = []
                for landmark in hand_landmarks.landmark:
                    landmarks.extend([landmark.x, landmark.y, landmark.z])

                # Convertir los landmarks en un array numpy y hacer la predicción
                landmarks = np.array(landmarks).reshape(1, -1)
                prediction = model.predict(landmarks)
                predicted_label = labels[np.argmax(prediction)]

                # Mostrar la letra/gesto predicho en la ventana
                cv2.putText(image, predicted_label, (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

        cv2.imshow('Tracking', image)

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
