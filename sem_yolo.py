import cv2
import pytesseract
import re

# Variável global para armazenar o texto da placa
ultimo_texto_placa = "Nenhuma placa detectada"

# Configurar o feed da webcam
camera_index = 0  # Índice da webcam (normalmente 0 para a câmera padrão)
cap = cv2.VideoCapture(camera_index)

if not cap.isOpened():
    print("Erro ao acessar a webcam.")
    exit()

# Criar uma janela para exibir as detecções
cv2.namedWindow("Detecção de Placas", cv2.WINDOW_NORMAL)

def processar_contornos(frame):
    # Pré-processamento
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, frame_thresh = cv2.threshold(frame_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Detectar contornos
    contornos, _ = cv2.findContours(frame_thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    possiveis_placas = []

    for contorno in contornos:
        perimetro = cv2.arcLength(contorno, True)
        aprox = cv2.approxPolyDP(contorno, 0.02 * perimetro, True)
        area = cv2.contourArea(contorno)
        x, y, w, h = cv2.boundingRect(contorno)

        # Filtrar por proporção e tamanho
        aspect_ratio = w / h
        if not (2.0 <= aspect_ratio <= 5.0):  # Ajuste para proporções típicas de placas
            continue
        if area < 5000 or area > 50000:  # Ajuste para áreas realistas de placas
            continue

        # Verificar se o contorno é quase retangular
        if len(aprox) == 4:  # Apenas considerar quadriláteros
            cv2.drawContours(frame, [aprox], -1, (0, 255, 0), 2)
            imagem_recortada = frame[y:y + h, x:x + w]
            imagem_recortada_cinza = cv2.cvtColor(imagem_recortada, cv2.COLOR_BGR2GRAY)
            _, imagem_recortada_limiarizada = cv2.threshold(imagem_recortada_cinza, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            imagem_processada = cv2.morphologyEx(imagem_recortada_limiarizada, cv2.MORPH_CLOSE, kernel)

            possiveis_placas.append((imagem_recortada, imagem_processada, (x, y, w, h)))

    return possiveis_placas

while True:
    ret, frame = cap.read()
    if not ret:
        print("Falha ao capturar frame da webcam.")
        break

    placas_detectadas = processar_contornos(frame)
    detected = False

    for placa_original, placa_processada, (x, y, w, h) in placas_detectadas:
        # Realizar OCR na região recortada
        texto_placa = pytesseract.image_to_string(placa_processada, config="--psm 7")

        # Processar texto para corresponder ao formato de placas
        texto_placa = ''.join(e for e in texto_placa if e.isalnum())
        texto_placa = texto_placa.replace("O", "0").replace("I", "1").strip()

        # Regex para verificar formato de placas do Brasil (AAA0A00)
        texto_extraido = re.search(r'^[A-Z]{3}[0-9][A-Z][0-9]{2}$', texto_placa)

        if texto_extraido:
            ultimo_texto_placa = texto_extraido.group()
            detected = True

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, ultimo_texto_placa, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    if not detected:
        cv2.putText(frame, "Nenhuma placa detectada", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Detecção de Placas", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
