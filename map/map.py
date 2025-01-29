import cv2
import numpy as np
import matplotlib.pyplot as plt

class CourtDetector:
    def __init__(self, image_path):
        self.image = cv2.imread(image_path)
        self.gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        self.edges = cv2.Canny(self.gray, 50, 150)
        self.drawing_key_points = [0] * 28
    
    def detect_lines(self):
        return cv2.HoughLinesP(self.edges, 1, np.pi / 180, threshold=100, minLineLength=50, maxLineGap=10)
    
    def find_intersections(self, lines):
        intersections = []
        for i in range(len(lines)):
            for j in range(i + 1, len(lines)):
                x1, y1, x2, y2 = lines[i][0]
                x3, y3, x4, y4 = lines[j][0]
                denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
                if denom == 0:
                    continue
                px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denom
                py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denom
                if 0 <= px <= self.image.shape[1] and 0 <= py <= self.image.shape[0]:
                    intersections.append((int(px), int(py)))
        return sorted(set(intersections), key=lambda p: (p[1], p[0]))
    
    def set_court_drawing_key_points(self):
        intersections = self.find_intersections(self.detect_lines())
        if len(intersections) >= 14:
            points = intersections[:14]
        else:
            points = intersections
        
        for i, (x, y) in enumerate(points):
            self.drawing_key_points[i * 2] = x
            self.drawing_key_points[i * 2 + 1] = y
    
    def draw_and_label_points(self):
        for i, (px, py) in enumerate(zip(self.drawing_key_points[::2], self.drawing_key_points[1::2])):
            cv2.circle(self.image, (px, py), 5, (0, 0, 255), -1)
            cv2.putText(self.image, str(i), (px + 5, py - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        plt.figure(figsize=(10, 5))
        plt.imshow(cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB))
        plt.title("Points détectés avec IDs")
        plt.axis("off")
        plt.show()
    
    def get_key_points(self):
        key_points_dict = {}
        for i in range(14):
            key_points_dict[f"point_{i}"] = (self.drawing_key_points[i * 2], self.drawing_key_points[i * 2 + 1])
        return key_points_dict

# Exemple d'utilisation
detector = CourtDetector(r"C:\Users\HP\OneDrive\Desktop\system_spark\videos\tactic_map.JPG")
detector.set_court_drawing_key_points()
detector.draw_and_label_points()
key_points = detector.get_key_points()
print("Points détectés:", key_points)
