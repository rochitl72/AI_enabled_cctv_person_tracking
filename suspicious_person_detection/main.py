import cv2
import torch
import face_recognition
import numpy as np
import sqlite3
import os
from ultralytics import YOLO

# ✅ Load YOLOv8 model
model = YOLO("yolov8n.pt")

# ✅ Load known faces from the 'faces/' directory
known_face_encodings = []
known_face_names = []
face_dir = "faces"

for filename in os.listdir(face_dir):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        image_path = os.path.join(face_dir, filename)
        image = face_recognition.load_image_file(image_path)
        encoding = face_recognition.face_encodings(image)
        if encoding:
            known_face_encodings.append(encoding[0])
            known_face_names.append(os.path.splitext(filename)[0])  # Extract name from filename

# ✅ Open video file
cap = cv2.VideoCapture("input/input.mp4")

if not cap.isOpened():
    print("Error: Could not open video file.")
    exit()

# ✅ Define video writer
output_path = "output/output.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
out = cv2.VideoWriter(output_path, fourcc, cap.get(5), (frame_width, frame_height))

# ✅ Connect to SQLite database
conn = sqlite3.connect("tracking_data.db")
cursor = conn.cursor()

# ✅ Ensure table exists
cursor.execute("""
CREATE TABLE IF NOT EXISTS tracked_people (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
)
""")
conn.commit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # ✅ Run YOLOv8 inference for person detection
    results = model(frame)

    persons = []
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get bounding box coordinates
            confidence = box.conf[0].item()
            class_id = int(box.cls[0])

            if class_id == 0:  # Only detect people
                persons.append((x1, y1, x2, y2))

    # ✅ Face recognition
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    face_names = []

    for face_encoding, (top, right, bottom, left) in zip(face_encodings, face_locations):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.5)
        name = "Unknown"

        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances) if len(face_distances) > 0 else None
        if best_match_index is not None and matches[best_match_index]:
            name = known_face_names[best_match_index]

        face_names.append((left, top, right, bottom, name))

        # Draw bounding box around face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 255), 2)  # Yellow for face
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    # ✅ Draw person detection boxes & associate faces
    for (x1, y1, x2, y2) in persons:
        detected_name = "Unknown"

        # Assign recognized face to detected person
        for (fx1, fy1, fx2, fy2, name) in face_names:
            if x1 < fx1 < x2 and y1 < fy1 < y2:  # If face is inside bounding box
                detected_name = name
                break

        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Blue for person detection
        cv2.putText(frame, detected_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Insert tracking data into SQLite database
        cursor.execute("INSERT INTO tracked_people (name) VALUES (?)", (detected_name,))
        conn.commit()

    # ✅ Write frame to output video
    out.write(frame)

# ✅ Release resources
cap.release()
out.release()
conn.close()
cv2.destroyAllWindows()
