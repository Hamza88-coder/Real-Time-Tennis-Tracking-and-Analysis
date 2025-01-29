from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StringType
import base64
import cv2
import numpy as np

from trackers import PlayerTracker, BallTracker
from mongoDB import FaceRecognizer
import findspark
from court_line_detector import CourtLineDetector
from mini_court import MiniCourt

findspark.init()

# Initialiser SparkSession
spark = SparkSession.builder \
    .appName("Kafka Spark Structured Streaming App") \
    .master("local[*]") \
    .config("spark.jars.packages", 
            "org.apache.spark:spark-sql-kafka-0-10_2.12:3.4.4"
            ) \
    .config("spark.executor.memory", "4g") \
    .config("spark.executor.cores", "4") \
    .config("spark.driver.memory", "4g") \
    .config("spark.hadoop.io.native.lib.available", "false") \
    .getOrCreate()

# Configurer les trackers et détecteurs
player_tracker = PlayerTracker(model_path=r'C:\Users\HP\OneDrive\Desktop\Tennis_analysis\bigData\yolov8x.pt')
ball_tracker = BallTracker(model_path=r'C:\Users\HP\OneDrive\Desktop\Tennis_analysis\bigData\models\best.pt')
court_line_detector = CourtLineDetector(model_path=r'C:\Users\HP\OneDrive\Desktop\Tennis_analysis\bigData\models\keypoints_model.pth')
mini_court = MiniCourt()
output_video_frames = []
firstly=True


# Configuration MongoDB
MONGO_URI = 'mongodb://root:example@localhost:27017'
DB_NAME = 'face_recognition_db'
ENCODINGS_COLLECTION = 'encodings'
recognizer = FaceRecognizer(MONGO_URI, DB_NAME, ENCODINGS_COLLECTION)

# Fonction pour décoder une image base64
def decode_base64_image(base64_str):
    img_data = base64.b64decode(base64_str)
    np_arr = np.frombuffer(img_data, dtype=np.uint8)
    return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

# Définir le schéma des messages Kafka
schema = StructType() \
    .add("camera_id", StringType()) \
    .add("data", StringType())  # Image encodée en base64

# Lecture du flux Kafka
df = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "localhost:29092") \
    .option("subscribe", "camera_streams") \
    .load()

# Transformation du flux Kafka en DataFrame avec schéma
df_parsed = df.selectExpr("CAST(value AS STRING)") \
    .selectExpr("from_json(value, 'camera_id STRING, data STRING') AS json") \
    .select("json.camera_id", "json.data")

# Tampon pour accumuler les frames
frame_buffer = []
court_keypoints = []

# Fonction de traitement pour Spark
def process_row(row):
    global output_video_frames
    global frame_buffer
    global firstly
    global court_keypoints
    camera_id = row.camera_id
    frame_data = row.data

    # Décoder l'image
    frame = decode_base64_image(frame_data)

    # Ajouter la frame au tampon
    frame_buffer.append(frame)
    if firstly:
          print("keypoints-------------------------------------------")
          
          court_keypoints = court_line_detector.predict(frame_buffer[0])
          firstly=False
    # Vérifier si 10 frames sont accumulées
    if len(frame_buffer) == 10:
        # Traiter les 10 frames
            print(f"Traitement de {len(frame_buffer)} frames pour la caméra {camera_id}")

        # Exemple de traitement avec les modèles
        
            detections = player_tracker.detect_frames(frame_buffer)
            detections=player_tracker.choose_and_filter_players(court_keypoints, detections)
            detection_ball = ball_tracker.detect_frames(frame_buffer)
            detection_ball = ball_tracker.interpolate_ball_positions(detection_ball)

            if len(detection_ball)>0:
               adjusments=mini_court.convert_bounding_boxes_to_mini_court_coordinates(detections, detection_ball, court_keypoints)
            
            # Dessiner les résultats
            frame_buffer = player_tracker.draw_bboxes(frame, detections)
            frame_buffer = ball_tracker.draw_bboxes(frame, detection_ball)
            output_video_frames.append(frame_buffer)
           

        # Vider le tampon après traitement
            frame_buffer = []

# Appliquer le traitement ligne par ligne
query = df_parsed.writeStream \
    .foreach(process_row) \
    .start()

query.awaitTermination()
