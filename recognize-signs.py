import cv2
import mediapipe as mp
import joblib
import numpy as np

# Cargar modelo entrenado
clf = joblib.load("sign_language_model.pkl")

# Inicializar MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Dibujar mano
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Extraer keypoints
                keypoints = []
                for lm in hand_landmarks.landmark:
                    keypoints.extend([lm.x, lm.y, lm.z])

                if len(keypoints) == 63:
                    # Clasificar
                    X = np.array(keypoints).reshape(1, -1)
                    prediction = clf.predict(X)[0]
                    confidence = np.max(clf.predict_proba(X))

                    # Mostrar predicción
                    cv2.putText(image, f"{prediction} ({confidence*100:.1f}%)", (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

        cv2.imshow("Reconocimiento de señas", image)
        if cv2.waitKey(10) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()