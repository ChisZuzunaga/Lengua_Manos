import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands    #Acceso al módulo de seguimiento de manos de mediapipe
mp_drawing = mp.solutions.drawing_utils    #Acceso a las utilidades de dibujo, para dibujar los puntos y conexiones de las manos

cap = cv2.VideoCapture(0)    #Se inicia la captura de video desde la cámara, el parámetro 0, se refiere a la cámara predeterminada del dispositivo

#Se inicia el objeto hands de mediapipe, el uso de with garantiza que se liberen los recursos al finalizar
#max_num_hands define que se detectaran maximo 2 manos
#min_detection confidence, es el valor minimo de confianza para considerar que una mano ha sido detectada
#min_tracking_confidence, el valor minimo de confianza para continuar rastreando la mano una vez detectada
with mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    while cap.isOpened():       #Mantiene un bucle continuo mientras la cámara esté activa, Dentro de este bucle es donde se procesa cada cuadro de vídeo
        success, frame = cap.read()        #Captura el siguiente cuadro de video de la camara, es True si fue éxitoso y frame contiene la imagen del cuadro
        if not success:
            print("Error al acceder a la cámara.")
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)        #Convierte el cuadro de color BGR (formato de OpenCV) a RGB (formato de MediaPipe)
        results = hands.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)        #Convierte la imagen de color BGR (formato de OpenCV) a RGB (formato de MediaPipe)
        
        if results.multi_hand_landmarks:        #Verifica si se detectaron las manos dentro del cuadro actual
            for hand_landmarks in results.multi_hand_landmarks:     #Itera sobre cada mano detectada
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)     #Dibuja los puntos clave y las conexiones de la mano detectada en la imagen

        cv2.imshow('Tracking', image)    #Muestra la imagen procesada en una ventana de OpenCV

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

cap.release()   #Libera la cámara y detiene la captura de video
cv2.destroyAllWindows()   #Cierra todas las ventanas de OpenCV
