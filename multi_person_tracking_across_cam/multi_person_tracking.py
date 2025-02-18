import cv2
import os
import numpy as np
import torch
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from tqdm import tqdm
from insightface.app import FaceAnalysis



import torch

# Check if MPS is available
if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

print(f"Using device: {device}")



# Device selection
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Load YOLOv8 model
yolo_model = YOLO("yolov8n.pt")  # Use 'yolov8s.pt' for better accuracy

# Initialize DeepSORT tracker
tracker = DeepSort(max_age=30)

# Initialize InsightFace (buffalo_l) model for facial recognition
facenet_model = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider' if device == 'cuda' else 'CPUExecutionProvider'])
facenet_model.prepare(ctx_id=0, det_size=(640, 640))

# Database to store known face embeddings and IDs
face_db = {}

# Define input and output directories
input_dir = "videos/"
output_dir = "output/"
os.makedirs(output_dir, exist_ok=True)

# Clothing color recognition helper function
def extract_clothing_color(frame, bbox):
    """Extracts the dominant color of the person's clothing."""
    x, y, w, h = bbox
    person_roi = frame[int(y+h*0.4):int(y+h*0.9), int(x):int(x+w)]  # Focus on the torso
    if person_roi.size == 0:
        return (0, 0, 0)  # Return black if no valid region
    avg_color = np.mean(person_roi, axis=(0, 1))
    return tuple(map(int, avg_color))

# Function to recognize faces using InsightFace
def recognize_face(face_img):
    """Extracts face embeddings and matches against stored embeddings."""
    face_results = facenet_model.get(face_img)
    
    if len(face_results) == 0:
        return None  # No face detected

    embedding = face_results[0].normed_embedding  # Extract face embedding

    # Compare against stored face embeddings
    min_dist = float("inf")
    best_match = None

    for face_id, stored_embedding in face_db.items():
        dist = np.linalg.norm(stored_embedding - embedding)
        if dist < 0.5 and dist < min_dist:  # Threshold for face similarity
            min_dist = dist
            best_match = face_id

    if best_match:
        return best_match
    else:
        # Store new face with a unique ID
        new_face_id = f"Person-{len(face_db) + 1}"
        face_db[new_face_id] = embedding
        return new_face_id

# Process each video in the input folder
for video_file in os.listdir(input_dir):
    if not video_file.endswith((".mp4", ".avi", ".mov")):
        continue
    
    video_path = os.path.join(input_dir, video_file)
    output_path = os.path.join(output_dir, f"output_{video_file}")
    
    cap = cv2.VideoCapture(video_path)
    width, height = int(cap.get(3)), int(cap.get(4))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    pbar = tqdm(total=frame_count, desc=f"Processing {video_file}")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = yolo_model(frame)[0]
        detections = []
        face_data = {}

        for result in results.boxes.data:
            x1, y1, x2, y2, conf, cls = result.cpu().numpy()
            if int(cls) == 0:  # Only detect people (YOLO class 0)
                bbox = (int(x1), int(y1), int(x2-x1), int(y2-y1))
                detections.append((bbox, conf, None))  # No feature data for DeepSORT

                # Extract face if available
                face_roi = frame[int(y1):int(y1 + (y2-y1)/3), int(x1):int(x2)]
                if face_roi.size > 0:
                    face_id = recognize_face(face_roi)
                    if face_id:
                        face_data[bbox] = face_id
                    else:
                        color = extract_clothing_color(frame, bbox)
                        face_data[bbox] = f"Color-{color}"

        # Update tracker with detections
        tracks = tracker.update_tracks(detections, frame=frame)

        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            ltrb = track.to_ltrb()
            x1, y1, x2, y2 = map(int, ltrb)
            label = face_data.get((x1, y1, x2-x1, y2-y1), f"ID-{track_id}")

            # Draw bounding box and ID
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        out.write(frame)
        pbar.update(1)

    cap.release()
    out.release()
    pbar.close()

print("Processing complete. Output saved in 'output/' directory.")
