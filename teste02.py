import cv2
from ultralytics import YOLO
import pytesseract
import re
import numpy as np
from collections import Counter

# Carregar o modelo YOLO treinado (substitua pelo caminho do modelo treinado)
model_path = "Yolo-tutov2/Placas-dec.v4i.yolov11/train/runs/detect/train/weights/best.pt"
model = YOLO(model_path)

# Variável global para armazenar o texto da placa
ultimo_texto_placa = "Nenhuma placa detectada"
historico_textos = []  # Buffer para histórico de textos detectados
buffer_size = 10  # Tamanho do buffer

# Configurar o feed da webcam
camera_index = 0  # Índice da webcam (normalmente 0 para a câmera padrão)
cap = cv2.VideoCapture(camera_index)
if not cap.isOpened():
    print("Erro ao acessar a webcam.")
    exit()

# Criar uma janela para exibir as detecções
cv2.namedWindow("Detecção de Placas", cv2.WINDOW_NORMAL)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Falha ao capturar frame da webcam.")
        break

    # Convertendo a imagem para o formato esperado pelo modelo (OpenCV usa BGR, enquanto YOLO usa RGB)
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Fazer a inferência usando o modelo YOLO
    results = model(img, conf=0.5, iou=0.4)

    # Limiar de confiança (ajuste conforme necessário, exemplo 0.5)
    confidence_threshold = 0.5

    # Acessando as caixas delimitadoras
    detections = results[0].boxes

    detected = False  # Flag para verificar se uma placa foi detectada

    for box in detections:
        conf = box.conf[0].cpu().numpy()  # Confiança da detecção
        if conf < confidence_threshold:
            continue  # Ignorar detecções com baixa confiança

        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)  # Coordenadas (x1, y1, x2, y2)
        cls = int(box.cls[0].cpu().numpy())  # Converter cls para um inteiro

        # Filtrando a classe que representa "placa"
        if cls in model.names:  # Verifica se a classe detectada está nas classes treinadas
            label = f"{model.names[cls]} {conf:.2f}"

            placa = frame[y1:y2, x1:x2]

            # Melhorar a imagem para OCR
            placa_gray = cv2.cvtColor(placa, cv2.COLOR_BGR2GRAY)  # Converter para escala de cinza

            # Aplicar CLAHE para melhorar o contraste
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            placa_contrast = clahe.apply(placa_gray)

            # Redimensionar a imagem da placa
            placa_resized = cv2.resize(placa_contrast, (300, 100))

            # Realizar OCR na região recortada
            config = "--psm 11"
            texto_placa = pytesseract.image_to_string(placa_resized, config=config)

            # Limpeza do texto reconhecido
            texto_placa = texto_placa.replace("O", "0").replace("I", "1").strip()

            # Verificar padrão específico (exemplo de placa brasileira)
            texto_extraido = re.search(r'[A-Z]{3}\d[A-Z]\d{2}', texto_placa)

            if texto_extraido:
                texto_valido = texto_extraido.group()
                historico_textos.append(texto_valido)
                if len(historico_textos) > buffer_size:
                    historico_textos.pop(0)

                # Determinar o texto mais frequente no buffer
                ultimo_texto_placa = Counter(historico_textos).most_common(1)[0][0]

            # Exibir o texto reconhecido na janela de detecção
            if conf >= 0.7:
                cv2.putText(frame, f"Placa: {ultimo_texto_placa}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                detected = True

            # Desenhar a caixa delimitadora e o rótulo
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Se não detectar nenhuma placa, mostrar a mensagem "Nenhuma placa detectada"
    if not detected:
        cv2.putText(frame, "Nenhuma placa detectada", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    # Mostrar a imagem com as detecções
    cv2.imshow("Detecção de Placas", frame)

    # Parar com a tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print(ultimo_texto_placa)
        break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()
