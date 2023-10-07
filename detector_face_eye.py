import cv2

# Cargar el clasificador de rostros
face_cascade = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('data/haarcascade_eye.xml')

# Abrir la cámara
cap = cv2.VideoCapture(0)

# Umbral para determinar si el ojo está cerrado
eye_closed_threshold = 12

while True:
    # Leer un fotograma de la cámara
    _, frame = cap.read()

    # Convertir la imagen a escala de grises
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detectar rostros en la imagen en escala de grises
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        # Dibujar un rectángulo alrededor de la cara detectada
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Obtener la región de interés (ROI) para ojos
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        # Detectar ojos en la región de la cara
        eyes = eye_cascade.detectMultiScale(roi_gray)

        for (ex, ey, ew, eh) in eyes:
            eye_roi = roi_gray[ey:ey + eh, ex:ex + ew]
            _, eye_thresh = cv2.threshold(eye_roi, 30, 255, cv2.THRESH_BINARY)
            eye_ratio = cv2.countNonZero(eye_thresh) / (ew * eh)

            # Dibujar un rectángulo alrededor del ojo
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)

            # Mostrar mensaje si el ojo está cerrado
            if eye_ratio < eye_closed_threshold:
                cv2.putText(frame, 'Ojo Cerrado', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    # Mostrar la imagen con los rectángulos
    cv2.imshow('Detector de cara', frame)

    # Esperar la pulsación de una tecla (tecla 'Esc' para salir)
    k = cv2.waitKey(30)
    if k == 27:
        break

# Liberar la captura y cerrar todas las ventanas
cap.release()
cv2.destroyAllWindows()