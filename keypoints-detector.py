import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convertir imagen BGR (OpenCV) a RGB (MediaPipe)
        results = hands.process(image)  # Procesar la imagen con MediaPipe
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Volver la imagen a BGR para OpenCV (para visualización)

        if results.multi_hand_landmarks:    # Si se detectó alguna mano
            for hand_landmarks in results.multi_hand_landmarks:
                # Dibujamos los puntos clave en la imagen
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Extraemos los 21 puntos
                keypoints = []
                for lm in hand_landmarks.landmark:
                    keypoints.extend([lm.x, lm.y, lm.z])     # 63 valores (21 puntos × 3)
                
                print("Keypoints: ", keypoints)

            cv2.imshow('Mano', image)
            if cv2.waitKey(5) & 0xFF==27:
                break

cap.release()
cv2.destroyAllWindows()
