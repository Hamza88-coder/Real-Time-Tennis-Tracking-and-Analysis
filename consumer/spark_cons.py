from kafka import KafkaConsumer
import base64
import cv2
import numpy as np
import json
from ultralytics import YOLO  # Assurez-vous d'avoir installé ultralytics (pip install ultralytics)

# Configuration du topic Kafka
TOPIC = "camera_streams"

# Initialisation du consommateur Kafka
consumer = KafkaConsumer(
    TOPIC,
    bootstrap_servers=['localhost:29092'],
    value_deserializer=lambda x: json.loads(x.decode('utf-8')),
    auto_offset_reset='earliest',  # Commence à lire à partir du début des messages
    group_id='camera-consumer-group',  # Groupe de consommateurs
    enable_auto_commit=True  # Les messages sont marqués comme lus automatiquement
)

# Charger le modèle YOLO
model = YOLO('yolov8x.pt')  # YOLOv8 Nano pour des performances rapides

# Fonction d'inférence YOLO
def apply_yolo(frame):
    """
    Applique YOLO sur une frame pour détecter des objets.
    :param frame: Image en format numpy array.
    :return: Liste des détections.
    """
    results = model(frame)  # Appliquer YOLO
    detections = []
    for result in results:
        for box in result.boxes:
            detections.append({
                "class": result.names[int(box.cls)],
                "confidence": float(box.conf),
                "bbox": box.xyxy.tolist()  # Coordonnées [x1, y1, x2, y2]
            })
    return detections

# Fonction pour décoder une image base64 en numpy array
def decode_base64_image(base64_str):
    """
    Décoder une image en base64 vers un tableau numpy.
    :param base64_str: Image encodée en base64.
    :return: Image en format numpy array.
    """
    img_data = base64.b64decode(base64_str)
    np_arr = np.frombuffer(img_data, dtype=np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    return frame

# Consommation des messages et affichage des images
if __name__ == "__main__":
    print("Démarrage du consommateur Kafka pour le topic 'camera_streams'...")

    try:
        while True:  # Boucle principale pour maintenir le programme actif
            for message in consumer:  # Parcourir les messages Kafka
                print('Nouveau message reçu')
                
                # Récupérer et analyser le message JSON
                message_value = message.value
                camera_id = message_value.get("camera_id")
                frame_data = message_value.get("data")  # Image encodée en base64

                if not frame_data:
                    print(f"Aucune donnée reçue pour la caméra {camera_id}.")
                    continue

                # Décoder l'image
                frame = decode_base64_image(frame_data)

                # Appliquer YOLO sur la frame
                detections = apply_yolo(frame)
                print(f"Caméra {camera_id}: {len(detections)} objets détectés.")

                # Afficher la frame avec les détections
                for detection in detections:
                    bbox = detection["bbox"]
                    if isinstance(bbox, list) and len(bbox) == 1 and isinstance(bbox[0], list):
                        bbox = bbox[0]  # Extraire la liste intérieure
                    if isinstance(bbox, np.ndarray):  # Si bbox est un tableau NumPy
                         bbox = bbox.tolist()
                    x1, y1, x2, y2 = map(int, bbox)
                    label = f"{detection['class']} ({detection['confidence']:.2f})"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Création d'une fenêtre pour la caméra
                window_name = f"Camera {camera_id} - Stream"
                cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)  # Permet le redimensionnement
                cv2.imshow(window_name, frame)

                # Gestion des touches pour arrêter le programme
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("Interruption demandée par l'utilisateur.")
                    break

    except KeyboardInterrupt:
        print("Arrêt du consommateur.")

    finally:
        # Fermer les fenêtres OpenCV avant de quitter
        cv2.destroyAllWindows()
