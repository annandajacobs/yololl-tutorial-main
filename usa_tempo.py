import cv2
from ultralytics import YOLO
import pytesseract
import re
import time

# Carregar o modelo YOLO treinado
model_path = "Yolo-tutov2/Placas-dec.v4i.yolov11/train/runs/detect/train/weights/best.pt"
model = YOLO(model_path)

# Variável para rastrear placas detectadas e seu tempo de permanência
rastreamento_placas = {}

# Tempo mínimo em segundos para considerar que uma placa esteve presente
tempo_minimo_presenca = 5  # 5 segundos

# Intervalo máximo sem detecção antes de remover uma placa
tempo_limite_sem_deteccao = 30  # 3 segundos

# Configurar o feed da webcam
camera_index = 0
cap = cv2.VideoCapture(camera_index)
if not cap.isOpened():
    print("Erro ao acessar a webcam.")
    exit()

cv2.namedWindow("Detecção de Placas", cv2.WINDOW_NORMAL)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Falha ao capturar frame da webcam.")
        break

    # Tempo atual em segundos
    tempo_atual = time.time()

    # Converter imagem para o formato esperado pelo YOLO
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = model(img, conf=0.5, iou=0.4)

    # Acessando caixas delimitadoras
    detections = results[0].boxes
    detected_placas = set()  # Para evitar múltiplas detecções da mesma placa no mesmo frame

    for box in detections:
        conf = box.conf[0].cpu().numpy()
        if conf < 0.5:
            continue

        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
        cls = int(box.cls[0].cpu().numpy())

        if cls in model.names:
            placa = frame[y1:y2, x1:x2]
            placa_gray = cv2.cvtColor(placa, cv2.COLOR_BGR2GRAY)
            _, placa_thresh = cv2.threshold(placa_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # OCR para extrair texto da placa
            texto_placa = pytesseract.image_to_string(placa_thresh, config="--psm 11")
            texto_placa = ''.join(e for e in texto_placa if e.isdigit() or e.isupper())
            texto_extraido = re.search(r'\w{3}\d{1}\w{1}\d{2}', texto_placa)

            if texto_extraido:
                texto_detectado = texto_extraido.group()

                # Atualizar rastreamento de placas
                if texto_detectado not in rastreamento_placas:
                    rastreamento_placas[texto_detectado] = {"inicio": tempo_atual, "ultimo": tempo_atual}
                else:
                    rastreamento_placas[texto_detectado]["ultimo"] = tempo_atual

                detected_placas.add(texto_detectado)

                # Exibir placa no frame
                cv2.putText(frame, f"Placa: {texto_detectado}", (x1, y2 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    # Remover placas que não foram detectadas recentemente
    placas_para_remover = []
    for placa, tempos in rastreamento_placas.items():
        if placa not in detected_placas and (tempo_atual - tempos["ultimo"]) > tempo_limite_sem_deteccao:
            placas_para_remover.append(placa)

    for placa in placas_para_remover:
        rastreamento_placas.pop(placa)

    # Mostrar a mensagem "Nenhuma placa detectada"
    if not detected_placas:
        cv2.putText(frame, "Nenhuma placa detectada", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

    # Exibir a imagem
    cv2.imshow("Detecção de Placas", frame)

    # Pressione 'q' para sair
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Após o loop, exibir placas que permaneceram pelo tempo mínimo
print("Placas que permaneceram pelo tempo mínimo:")
for placa, tempos in rastreamento_placas.items():
    tempo_presente = tempos["ultimo"] - tempos["inicio"]
    if tempo_presente >= tempo_minimo_presenca:
        print(f"Placa: {placa} - Tempo: {tempo_presente:.2f} segundos")

# Liberar recursos
cap.release()
cv2.destroyAllWindows()

