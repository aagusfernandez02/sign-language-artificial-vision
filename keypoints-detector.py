import os
import cv2
import csv
import string
import mediapipe as mp


mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

# Crear el archivo CSV si no existe
csv_filename = 'dataset.csv'
if not os.path.exists(csv_filename):
    with open(csv_filename, mode="w", newline='') as f:
        writer = csv.writer(f)
        headers = [f'x{i}' for i in range(21)] + [f'y{i}' for i in range(21)] + [f'z{i}' for i in range(21)] + ['label']
        writer.writerow(headers)


with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            continue

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convertir imagen BGR (OpenCV) a RGB (MediaPipe)
        results = hands.process(image)  # Procesar la imagen con MediaPipe
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Volver la imagen a BGR para OpenCV (para visualización)

        if results.multi_hand_landmarks:    # Si se detectó alguna mano
            for hand_landmarks in results.multi_hand_landmarks:
                # Dibujamos los puntos clave en la imagen
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        cv2.imshow('Mano', image)   # Mostrar la imagen con keypoints
        
        key = cv2.waitKey(10) & 0xFF    # Capturar tecla presionada
        if key == 27:   # Salir del ciclo si presionan ESC (ASCII 27)
            break

        if chr(key).lower() in string.ascii_lowercase: # Capturo si presionan una tecla de a-z
            letra = chr(key).upper()
            print(f"Letra '{letra}' presionada")

            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]    # Tomo la primera o única mano detectada
                keypoints = []
                for lm in hand_landmarks.landmark:
                    keypoints.append(lm.x)
                    keypoints.append(lm.y)
                    keypoints.append(lm.z)
                
                with open(csv_filename, mode='a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(keypoints + [letra])
                
                print(f"Ejemplo guardado para la letra '{letra}'")
            else:
                print("No se detectó ninguna mano. Intentar nuevamente")


cap.release()
cv2.destroyAllWindows()
