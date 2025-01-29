import threading
import datetime
import cv2
from kafka import KafkaProducer
from json import dumps
import base64
import time  # Ajout du module time

# Configuration du topic Kafka
TOPIC = "camera_streams"

# Initialisation du producteur Kafka
producer = KafkaProducer(
    bootstrap_servers=['localhost:29092'],
    value_serializer=lambda x: dumps(x).encode('utf-8')
)

# Liste des chemins des vidéos (remplacez par vos propres vidéos)
video_paths = [
    r"C:\Users\HP\OneDrive\Desktop\system_spark\videos\input_video.mp4",
   
    
]

def simulate_camera(camera_id, video_path):
    """
    Fonction pour simuler la capture d'une vidéo et envoyer les données à Kafka.
    """
    # Charger la vidéo avec OpenCV
    camera = cv2.VideoCapture(video_path)
    if not camera.isOpened():
        print(f"Erreur : la vidéo pour la caméra {camera_id} n'a pas pu être chargée.")
        return

    try:
        while True:
            success, frame = camera.read()
            if not success:
                print(f"Fin de la vidéo pour la caméra {camera_id}.")
                break
            
           
            
            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                print(f"Erreur : échec d'encodage pour la caméra {camera_id}.")
                continue

            # Encodage en base64
            data = base64.b64encode(buffer).decode('utf-8')

            # Création du message JSON
            message = {
                "camera_id": camera_id,
                "date": str(datetime.date.today()),
                "time": datetime.datetime.now().strftime("%X"),
                "rows": frame.shape[0],  # Hauteur de l'image
                "cols": frame.shape[1],  # Largeur de l'image
                "data": data
            }

            # Envoi au topic Kafka
            producer.send(TOPIC, value=message)
            print(f"Caméra {camera_id}: image envoyée à Kafka.")

            # Ajout d'un délai de 2 secondes entre chaque envoi
          

    except Exception as e:
        print(f"Erreur dans la caméra {camera_id}: {str(e)}")
    finally:
        camera.release()
        print(f"Caméra {camera_id}: arrêtée.")

# Lancement de la simulation pour 5 vidéos
if __name__ == "__main__":
    print("Démarrage de la simulation pour 5 caméras...")
    threads = []

    # Créer et démarrer un thread pour chaque vidéo
    for cam_id, video_path in enumerate(video_paths, start=1):
        thread = threading.Thread(target=simulate_camera, args=(cam_id, video_path))
        threads.append(thread)
        thread.start()

    # Attendre que tous les threads terminent
    for thread in threads:
        thread.join()

    print("Simulation terminée.")