import cv2
from ultralytics import YOLO

# Carregar o modelo YOLO treinado (substitua pelo caminho do modelo treinado)
model_path = "runs/detect/train3/weights/best.pt"
model = YOLO(model_path)

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
    results = model(img)

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

            # Desenhar a caixa delimitadora e o rótulo
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            detected = True  # Marcar que uma placa foi detectada

    # Se não detectar nenhuma placa, mostrar a mensagem "Nenhuma placa detectada"
    if not detected:
        frame = cv2.putText(frame, "Nenhuma placa detectada", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    # Mostrar a imagem com as detecções
    cv2.imshow("Detecção de Placas", frame)

    # Parar com a tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()
