#Maria Luiza Henrique Barbosa
#Guilherme Vendruscolo Cangemi Silva
#Gabriel Francisco Ribeiro

import cv2
import numpy as np

# Dicionário de mapeamento de faixas de valores para palavras de cor, adicionado manualmente
color_ranges = {
    (2, 7): "Rosa",
    (8, 19): "Laranja",
    (11, 40): "Amarelo",
    (41, 70): "Verde",
    (71, 104): "Azul",
    (106, 109): "Azul",
    (111, 148): "Roxo",
    (149, 169): "Rosa",
    (171, 190): "Vermelho",
    (191, 210): "Amarelo",
    (211, 240): "Amarelo",
    (271, 300): "Marrom",
    (301, 330): "Ciano",
    (331, 360): "Magenta",
    (105,105): "Preto",
    (110,110): "Preto",
    (0, 1): "Preto",
    (161, 180): "Preto",
    (181, 200): "Preto",
    (201, 270): "Amarelo",
    (271, 290): "Marrom Claro"
}

def detect_color(frame):
    # Converte a imagem para o espaço de cores HSV
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Calcula o histograma da matiz
    hist = cv2.calcHist([hsv_frame], [0], None, [180], [0, 180])

    # Obtém o valor do bin com maior frequência (cor predominante)
    max_bin = np.argmax(hist)
    color_value = max_bin

    # Verifica a faixa de valor correspondente à cor predominante
    color_name = "Desconhecida"
    for color_range, name in color_ranges.items():
        if color_range[0] <= color_value <= color_range[1]:
            color_name = name
            break

    return color_value

# Inicializa a câmera
#Para usar com dispositivo de video secundario mude para 1
cap = cv2.VideoCapture(0)

while True:
    # Lê o frame da câmera
    ret, frame = cap.read()

    # Espelha o frame horizontalmente
    frame = cv2.flip(frame, 1)

    # Converte a imagem para o espaço de cores HSV
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define os intervalos de cor para detecção
    lower_range = np.array([0, 75, 0])
    upper_range = np.array([180, 255, 255])

    # Segmenta a cor de interesse usando máscara
    color_mask = cv2.inRange(hsv_frame, lower_range, upper_range)

    # Aplica operações de abertura e fechamento para reduzir ruídos
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    opened_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, kernel)
    closed_mask = cv2.morphologyEx(opened_mask, cv2.MORPH_CLOSE, kernel)
    # Encontra contornos na máscara
    contours, _ = cv2.findContours(closed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filtra contornos considerando a área
    filtered_contours = [contour for contour in contours if cv2.contourArea(contour) > 100]

    # Encontra o objeto de destaque (maior área)
    if len(filtered_contours) > 0:
        main_contour = max(filtered_contours, key=cv2.contourArea)
        
        # Calcula o retângulo delimitador do objeto de destaque
        x, y, w, h = cv2.boundingRect(main_contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Recorta a região do objeto de destaque
        roi = frame[y:y + h, x:x + w]

        # Detecta a cor predominante no objeto de destaque
        color = detect_color(roi)
        
        color_name = "Desconhecida"
        for color_range, name in color_ranges.items():
            if color_range[0] <= color <= color_range[1]:
                color_name = name
                break

        # Mostra o frame e a cor detectada
        cv2.putText(frame, "Cor: " + color_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)

    # Mostra o frame original
    cv2.imshow("Detecção de Cores e Contornos", frame)

    # Verifica se a tecla 'q' foi pressionada para sair
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# Libera a câmera e fecha as janelas
cap.release()
cv2.destroyAllWindows()
