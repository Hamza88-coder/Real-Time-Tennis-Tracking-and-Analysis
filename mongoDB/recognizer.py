import cv2
import face_recognition
import numpy as np
from pymongo import MongoClient

class FaceRecognizer:
    def __init__(self, mongo_uri, db_name, encodings_collection_name):
        """
        Initialise la connexion à MongoDB et charge les encodages des visages.
        """
        self.client = MongoClient(mongo_uri)
        self.db = self.client[db_name]
        self.encodings_collection = self.db[encodings_collection_name]

    def load_encodings_from_mongodb(self):
        """
        Charge les encodages des visages depuis MongoDB.
        """
        cursor = self.encodings_collection.find({})
        known_encodings = []
        player_ids = []

        for record in cursor:
            known_encodings.append(record["encoding"])
            player_ids.append(record["player_id"])

        if not known_encodings:
            raise ValueError("Aucun encodage trouvé dans la collection MongoDB.")

        return known_encodings, player_ids

    def identify_person(self, image):
        """
        Identifie une personne en comparant l'image donnée avec les encodages stockés.
        """
        # Convertir l'image en RGB pour face_recognition
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        


        face_encodings = face_recognition.face_encodings(img_rgb)
        print(len(face_encodings))

        if len(face_encodings) == 0:
            print("Aucun visage détecté dans l'image.")
            return None

        # Charger les encodages depuis MongoDB
        known_encodings, player_ids = self.load_encodings_from_mongodb()

        # Comparer les visages
        matches = face_recognition.compare_faces(known_encodings, face_encodings[0])
        distances = face_recognition.face_distance(known_encodings, face_encodings[0])

        # Identifier le visage avec la distance minimale si une correspondance est trouvée
        if any(matches):
            best_match_index = np.argmin(distances)
            identified_id = player_ids[best_match_index]
            print(f"Personne identifiée : {identified_id}")
            return identified_id
        else:
            print("Aucune correspondance trouvée.")
            return None


if __name__ == "__main__":
    # Configuration MongoDB
    MONGO_URI = 'mongodb://root:example@localhost:27017'
    DB_NAME = 'face_recognition_db'
    ENCODINGS_COLLECTION = 'encodings'

    # Chemin de la vidéo à identifier
    VIDEO_PATH = r"C:\Users\HP\OneDrive\Desktop\system_spark\videos\Novak Djokovic vs Carlos Alcaraz _ Quarter Final _ Australian Open 2025 Extended Highlights 🇦🇺(1080P_HD).mp4"

    # Charger la vidéo
    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        raise FileNotFoundError(f"Vidéo introuvable ou impossible à ouvrir : {VIDEO_PATH}")

    # Initialiser la classe de reconnaissance faciale
    recognizer = FaceRecognizer(MONGO_URI, DB_NAME, ENCODINGS_COLLECTION)

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Fin de la vidéo ou problème de lecture.")
            break

        # Identifier les personnes dans la frame
        identified_person = recognizer.identify_person(frame)

        

        # Quitter si la touche 'q' est pressée
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Libérer les ressources
    cap.release()
    cv2.destroyAllWindows()

