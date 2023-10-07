import cv2

# Cargar el clasificador de rostros
face_cascade = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')
# Abrir la cámara
cap = cv2.VideoCapture(0)

while True:
    # Leer un fotograma de la cámara
    _, CaraPersona = cap.read()
    
    # Convertir la imagen a escala de grises
    gray = cv2.cvtColor(CaraPersona, cv2.COLOR_BGR2GRAY)
    
    # Detectar caras en la imagen en escala de grises
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    # Dibujar rectángulos alrededor de las caras detectadas
    for (x, y, w, h) in faces:
        cv2.rectangle(CaraPersona, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # Mostrar la imagen con los rectángulos
    cv2.imshow('Detector de cara', CaraPersona)
    
    # Esperar la pulsación de una tecla (tecla 'Esc' para salir)
    k = cv2.waitKey(30)
    if k == 27:
        break

# Liberar la captura y cerrar todas las ventanas
cap.release()
cv2.destroyAllWindows()