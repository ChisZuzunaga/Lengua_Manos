import cv2
import mediapipe as mp
import csv

# Inicializar MediaPipe y OpenCV
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)

# Abrir archivo CSV para guardar los datos
with open('lengua_manos/data/gestures_data.csv', mode='w', newline='') as f:
    csv_writer = csv.writer(f)

    # Definir las columnas (landmarks) para 21 puntos por mano (x, y, z) por punto clave
    headers = []
    for i in range(21):
        headers += [f'x{i}', f'y{i}', f'z{i}']
    headers.append('label')  # Añadimos la etiqueta del gesto
    csv_writer.writerow(headers)

    # Usar MediaPipe para la detección de las manos
    with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.75, min_tracking_confidence=0.75) as hands:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                print("Error al acceder a la cámara.")
                break

            frame = cv2.resize(frame, (640, 480))
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

                    # Asignar la etiqueta (letra o gesto que estás haciendo)
                    label = input("Introduce la letra o gesto (q para salir): ")
                    if label == 'q':
                        cap.release()
                        cv2.destroyAllWindows()
                        exit()

                    # Guardar landmarks y etiqueta en el archivo CSV
                    landmarks.append(label)
                    csv_writer.writerow(landmarks)

            cv2.imshow('Tracking', image)

            if cv2.waitKey(200) & 0xFF == ord('q'):
                break

cap.release()
cv2.destroyAllWindows()
