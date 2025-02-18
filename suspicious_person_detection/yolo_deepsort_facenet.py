import cv2
import torch
import face_recognition
import numpy as np
import sqlite3
import os
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# ✅ Load YOLOv8 model
model = YOLO("yolov8n.pt")

# ✅ Initialize DeepSORT tracker
tracker = DeepSort(max_age=50, n_init=3, max_iou_distance=0.7)

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

# ✅ Open webcam (use your correct device index)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# ✅ Connect to SQLite database
conn = sqlite3.connect("tracking_data.db")
cursor = conn.cursor()

# ✅ Ensure table exists
cursor.execute("""
CREATE TABLE IF NOT EXISTS tracked_people (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    track_id INTEGER,
    name TEXT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
)
""")
conn.commit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # ✅ Run YOLOv8 inference
    results = model(frame)

    detections = []
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get bounding box coordinates
            confidence = box.conf[0].item()
            class_id = int(box.cls[0])

            if class_id == 0:  # Only track people
                detections.append(([x1, y1, x2, y2], confidence, "person"))

    # ✅ Update tracker with detected people
    tracks = tracker.update_tracks(detections, frame=frame)

    # ✅ Face recognition
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    face_names = {}  # Dictionary to store track_id → recognized name

    for face_encoding, (top, right, bottom, left) in zip(face_encodings, face_locations):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.5)
        name = "Unknown"

        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances) if len(face_distances) > 0 else None
        if best_match_index is not None and matches[best_match_index]:
            name = known_face_names[best_match_index]

        # Store recognized face name
        face_names[(left, top, right, bottom)] = name

        # Draw bounding box around face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 255), 2)  # Yellow for face
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    # ✅ Draw tracking results & insert into database
    for track in tracks:
        if not track.is_confirmed():
            continue
        
        track_id = track.track_id
        ltrb = track.to_ltrb()
        x1, y1, x2, y2 = map(int, ltrb)
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Blue for tracking
        cv2.putText(frame, f"ID {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Assign face name if found
        detected_name = "Unknown"
        for (fx1, fy1, fx2, fy2), name in face_names.items():
            if x1 < fx1 < x2 and y1 < fy1 < y2:  # If face is inside bounding box
                detected_name = name
                break

        # Insert tracking data into SQLite database
        cursor.execute("INSERT INTO tracked_people (track_id, name) VALUES (?, ?)", (track_id, detected_name))
        conn.commit()

    # ✅ Show webcam feed with tracking
    cv2.imshow("YOLOv8 + DeepSORT + Face Recognition + SQLite", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ✅ Release resources
cap.release()
conn.close()
cv2.destroyAllWindows()
