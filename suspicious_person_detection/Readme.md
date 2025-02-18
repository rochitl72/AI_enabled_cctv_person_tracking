# Face Recognition and Person Detection System

This project processes an input video (`input/input.mp4`), detects people using YOLOv8, recognizes faces using `face_recognition`, and stores tracking data in an SQLite database. The processed video is saved as `output/output.mp4`.

## Installation

### Requirements
Ensure you have the required dependencies installed:

```sh
pip install opencv-python torch ultralytics face-recognition numpy sqlite3
```

### Folder Structure
```
project/
│-- faces/            # Folder containing images of known people (JPG/PNG format)
│-- input/input.mp4   # Input video for processing
│-- output/output.mp4 # Output video with detections
│-- tracking_data.db  # SQLite database storing tracking data
│-- script.py         # Main Python script
```

## How It Works

1. **Load YOLOv8 Model**:
   - Uses `ultralytics.YOLO` to detect people in each frame.

2. **Face Recognition**:
   - Loads face images from the `faces/` directory.
   - Recognizes faces using `face_recognition` library.

3. **Processing the Video**:
   - Reads frames from `input/input.mp4`.
   - Runs YOLOv8 inference to detect people.
   - Recognizes faces and assigns names (or marks them as "Unknown").
   - Saves the processed video with bounding boxes and labels.

4. **Database Logging**:
   - Stores each detected person’s name (if recognized) in `tracking_data.db` with a timestamp.

## Running the Script

Run the Python script to process the input video:

```sh
python init_db.py
python main.py
```

## Output
- The processed video with bounding boxes around detected people and recognized faces will be saved in `output/output.mp4`.
- Recognized names will be stored in the SQLite database `tracking_data.db`.

## Notes
- Ensure `faces/` contains images of known individuals for better recognition accuracy.
- You can adjust the `tolerance` parameter in `face_recognition.compare_faces` to fine-tune face matching.
- Press `q` to quit video processing manually.

## Future Enhancements
- Multi-camera support.
- Improved tracking mechanisms.
- Cloud-based storage for detected data.

