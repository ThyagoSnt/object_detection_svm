import cv2
import numpy as np

# Função para atualizar os valores HSV
def update_hsv(x):
    pass

# Captura o vídeo
video_path = 'bases.mp4'  # Substitua pelo caminho do seu vídeo
cap = cv2.VideoCapture(video_path)

# Cria a janela e as trackbars para ajustar os valores HSV
cv2.namedWindow('HSV Adjustments')
cv2.createTrackbar('Hue Min', 'HSV Adjustments', 0, 179, update_hsv)
cv2.createTrackbar('Hue Max', 'HSV Adjustments', 179, 179, update_hsv)
cv2.createTrackbar('Sat Min', 'HSV Adjustments', 0, 255, update_hsv)
cv2.createTrackbar('Sat Max', 'HSV Adjustments', 255, 255, update_hsv)
cv2.createTrackbar('Val Min', 'HSV Adjustments', 0, 255, update_hsv)
cv2.createTrackbar('Val Max', 'HSV Adjustments', 255, 255, update_hsv)

while True:
    ret, frame = cap.read()
    if ret:
        frame = cv2.resize(frame, (640,480))

        # Se o vídeo terminar, reinicie-o
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        # Converte a imagem para o espaço de cor HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Obtém os valores atuais das trackbars
        h_min = cv2.getTrackbarPos('Hue Min', 'HSV Adjustments')
        h_max = cv2.getTrackbarPos('Hue Max', 'HSV Adjustments')
        s_min = cv2.getTrackbarPos('Sat Min', 'HSV Adjustments')
        s_max = cv2.getTrackbarPos('Sat Max', 'HSV Adjustments')
        v_min = cv2.getTrackbarPos('Val Min', 'HSV Adjustments')
        v_max = cv2.getTrackbarPos('Val Max', 'HSV Adjustments')

        # Define o intervalo de valores HSV
        lower_hsv = np.array([h_min, s_min, v_min])
        upper_hsv = np.array([h_max, s_max, v_max])

        # Cria uma máscara baseada nos valores definidos
        mask = cv2.inRange(hsv, lower_hsv, upper_hsv)
        result = cv2.bitwise_and(frame, frame, mask=mask)

        # Mostra o vídeo original e o resultado filtrado
        cv2.imshow('Original Video', frame)
        cv2.imshow('Filtered Video', result)

        # Sai do loop se a tecla 'q' for pressionada
        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

# Libera o vídeo e fecha as janelas
cap.release()
cv2.destroyAllWindows()
