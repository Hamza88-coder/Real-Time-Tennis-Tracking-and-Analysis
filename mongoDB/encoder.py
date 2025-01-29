import cv2
import face_recognition
import os
from pymongo import MongoClient
import numpy as np

class FaceEncoder:
    def __init__(self, mongo_uri, db_name, players_collection_name, encodings_collection_name):
        """
        Initialise la connexion à MongoDB et les collections.
        """
        self.client = MongoClient(mongo_uri)
        self.db = self.client[db_name]
        self.players_collection = self.db[players_collection_name]
        self.encodings_collection = self.db[encodings_collection_name]

    def load_images_from_folder(self, folder_path):
        """
        Charge les images depuis un dossier et retourne une liste d'images et leurs IDs.
        """
        if not os.path.exists(folder_path):
            raise FileNotFoundError(f"Le dossier spécifié n'existe pas : {folder_path}")

        img_list = []
        player_ids = []

        for filename in os.listdir(folder_path):
            img_path = os.path.join(folder_path, filename)
            img = cv2.imread(img_path)

            if img is None:
                print(f"Erreur : Impossible de lire l'image {filename}. Vérifiez le format.")
                continue

            # Convertir l'image en RGB (face_recognition utilise RGB)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
          
            cv2.namedWindow("window_name", cv2.WINDOW_NORMAL)  # Permet le redimensionnement
            cv2.imshow("window_name", img_rgb)

# Attendez qu'une touche soit pressée avant de fermer la fenêtre
            cv2.waitKey(0)  # 0 signifie attendre indéfiniment
            cv2.destroyAllWindows()
            img_list.append(img_rgb)

            # Utiliser le nom du fichier (sans extension) comme ID du joueur
            player_id = os.path.splitext(filename)[0]
            player_ids.append(player_id)

            # Enregistrer l'image dans MongoDB
            self.save_image_to_mongodb(player_id, filename, img_path)

        return img_list, player_ids

    def save_image_to_mongodb(self, player_id, image_name, image_path):
        """
        Enregistre une image dans la collection MongoDB des joueurs.
        """
        with open(image_path, 'rb') as image_file:
            encoded_image = image_file.read()

        self.players_collection.insert_one({
            "player_id": player_id,
            "image_name": image_name,
            "image_data": encoded_image  # Image encodée en binaire
        })
        print(f"Image {image_name} uploadée sur MongoDB")

    def encode_faces(self, img_list):
        """
        Encode les visages détectés dans une liste d'images.
        """
        encode_list = []
        for img in img_list:
            # Encoder les visages détectés dans l'image
            face_encodings = face_recognition.face_encodings(img)

            if len(face_encodings) > 0:
                encode_list.append(face_encodings[0].tolist())  # Convertir en liste pour MongoDB
            else:
                print("Aucun visage détecté dans une image.")

        return encode_list

    def save_encodings_to_mongodb(self, encodings, player_ids):
        """
        Enregistre les encodages des visages dans MongoDB.
        """
        for encoding, player_id in zip(encodings, player_ids):
            self.encodings_collection.insert_one({
                "player_id": player_id,
                "encoding": encoding
            })
        print("Encodings enregistrés dans MongoDB")

    def process_folder(self, folder_path):
        """
        Traite un dossier d'images : charge les images, encode les visages et enregistre les résultats.
        """
        # Charger les images et leurs IDs
        img_list, player_ids = self.load_images_from_folder(folder_path)

        if not img_list:
            print("Aucune image valide trouvée dans le dossier.")
            return

        # Encoder les visages détectés
        print("Encodage en cours...")
        encode_list = self.encode_faces(img_list)
        print(encode_list)

        if not encode_list:
            print("Aucun encodage trouvé. Vérifiez les images ou le dossier.")
            return

        # Enregistrer les encodages dans MongoDB
        self.save_encodings_to_mongodb(encode_list, player_ids)
        print("Encodage terminé.")


# Utilisation de la classe FaceEncoder
if __name__ == "__main__":
    # Configuration MongoDB
    MONGO_URI = 'mongodb://root:example@localhost:27017'
    DB_NAME = 'face_recognition_db'
    PLAYERS_COLLECTION = 'tennis_players'
    ENCODINGS_COLLECTION = 'encodings'

    # Chemin du dossier contenant les images des joueurs
    FOLDER_PATH = r"C:\Users\HP\OneDrive\Desktop\system_spark\mongoDB\Tennis_Images"

    # Initialiser et exécuter le processus
    face_encoder = FaceEncoder(MONGO_URI, DB_NAME, PLAYERS_COLLECTION, ENCODINGS_COLLECTION)
    face_encoder.process_folder(FOLDER_PATH)
