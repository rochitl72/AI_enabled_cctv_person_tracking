# Multi-Camera Person Tracking and Recognition System

## Overview
This project implements a **multi-camera person tracking and recognition system** using **YOLOv8** for person detection, **DeepSORT** for object tracking, and **InsightFace** for facial recognition. It processes multiple videos, detects people, tracks them across frames, and attempts face recognition. If a face is not recognized, the system tracks the person using clothing color.

## Features
- **Person Detection**: Utilizes **YOLOv8** to detect people in video frames.
- **Multi-Object Tracking**: Uses **DeepSORT** to track detected individuals.
- **Face Recognition**: Employs **InsightFace** to recognize and match faces.
- **Clothing-Based ID**: Assigns an ID based on clothing color if face recognition fails.
- **Multi-Camera Processing**: Processes multiple video files from an input directory.
- **Real-Time Processing Updates**: Displays progress using `tqdm`.
- **Automatic Video Output Generation**: Saves processed videos with bounding boxes and IDs.

## Requirements
Ensure you have the following dependencies installed:

```sh
pip install opencv-python numpy torch ultralytics tqdm insightface deep_sort_realtime
```

### Hardware Requirements
- A **MacBook Pro with M1 Pro** is recommended for optimal performance.
- Supports **CPU, GPU (CUDA), and Apple MPS (Metal Performance Shaders)**.

## Setup and Usage
### 1. Clone the Repository
```sh
git clone <repository_url>
cd <repository_folder>
```

### 2. Organize Input Videos
- Place video files in the `videos/` directory.
- Supported formats: `.mp4`, `.avi`, `.mov`.

### 3. Run the Tracker
```sh
python track_faces.py
```

### 4. Check the Output
Processed videos will be saved in the `output/` directory with bounding boxes and assigned IDs.

## File Structure
```
├── videos/             # Input video files
├── output/             # Processed video files with tracking results
├── track_faces.py      # Main script for tracking and recognition
├── README.md           # Project documentation
```

## How It Works
1. **YOLOv8** detects people in each video frame.
2. **DeepSORT** assigns tracking IDs to detected individuals.
3. **InsightFace** extracts and matches face embeddings.
4. If a face is unrecognized, the system assigns an ID based on **clothing color**.
5. Bounding boxes and IDs are overlaid onto the video frames.
6. Processed videos are saved to the `output/` directory.

## Device Selection
The script automatically detects available hardware and sets the appropriate device:
```python
if torch.backends.mps.is_available():
    device = "mps"  # Apple Silicon GPU
elif torch.cuda.is_available():
    device = "cuda"  # NVIDIA GPU
else:
    device = "cpu"  # Fallback to CPU
```

## Face Recognition Logic
- Extracts **face embeddings** using **InsightFace**.
- Matches against stored face embeddings using **cosine similarity**.
- If no match is found (threshold `0.5`), assigns a new unique ID.
- Stores embeddings for future recognition.

## Clothing Recognition Logic
- Extracts **dominant clothing color** from the torso region.
- Uses it as an alternative identifier if face recognition fails.

## Output Example
- Recognized face: `Person-1`
- Unrecognized person (clothing-based ID): `Color-(R,G,B)`

## Performance Considerations
- **YOLOv8n** (nano model) is used for speed; switch to **YOLOv8s** for better accuracy.
- Uses **DeepSORT's Kalman filter** for smoother tracking.
- Optimize `det_size=(640, 640)` in **InsightFace** for faster face detection.

## Future Improvements
- Implement **multi-camera synchronization** for real-time tracking.
- Integrate **behavior analysis** (e.g., fall detection, suspicious activity tracking).
- Add a **live streaming** feature for real-time CCTV surveillance.

## License
This project is open-source under the **MIT License**.

## Acknowledgments
- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- [DeepSORT](https://github.com/nwojke/deep_sort)
- [InsightFace](https://github.com/deepinsight/insightface)


