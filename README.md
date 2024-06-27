
https://github.com/Parshantladhar/Real-Time-Vehicle-Speed-Detection-using-YOLOv8_with-GUI/assets/120705862/5432f012-8539-4878-a204-bc99d01c9b77
# Real-Time Vehicle Speed Detection

This project is a real-time vehicle speed detection application using the YOLOv8 model. The application features a graphical user interface (GUI) built with PyQt5, which allows users to process video files or use real-time detection via a webcam. The application detects vehicles, tracks them, and calculates their speeds.

## Features

- **Real-time vehicle detection and speed calculation** using the YOLOv8 model.
- **GUI interface** for ease of use.
- **Support for video file processing** and real-time detection via webcam.
- **Visual output** with detected vehicles, tracking lines, and calculated speeds displayed on the video.

## Installation
### Requirements

- Python 3.8 or higher
- `PyQt5`
- `opencv-python`
- `pandas`
- `ultralytics`
- `torch`

### Steps

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/vehicle-speed-detection.git
   cd vehicle-speed-detection
   ```

2. **Create and activate a virtual environment (optional but recommended)**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install the required packages**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Download the YOLOv8 model** and place it in the project directory:
   - You can download the model from the [Ultralytics YOLOv8 repository](https://github.com/ultralytics/yolov8).

5. **Ensure you have the COCO class labels file** (`coco.txt`) in the project directory. You can download it from [COCO dataset](https://cocodataset.org/#home).

### Running the Application

1. **Run the application**:
   ```bash
   python gui.py
   ```

2. **Use the GUI** to select a video file or choose real-time detection via a webcam.

## Configuration

You can adjust the configuration settings in the `config.py` file:

- `YOLO_MODEL_PATH`: Path to the YOLO model file.
- `COCO_CLASSES_PATH`: Path to the COCO class labels file.
- `OUTPUT_VIDEO_PATH`: Path to the output video file.
- `CY1` and `CY2`: Coordinates for the lines used in vehicle tracking.
- `OFFSET`: Offset value for vehicle tracking.
- `VIDEO_WIDTH` and `VIDEO_HEIGHT`: Dimensions of the video frames.
- `VIDEO_FPS`: Frames per second for video capture.
- `DISTANCE`: Distance in meters for speed calculation.

## Project Structure

- `config.py`: Configuration file for paths and constants.
- `gui.py`: Main GUI application file.
- `tracker.py`: Vehicle tracking logic.
- `main.py`: (Optional) Additional main script.
- `requirements.txt`: List of Python dependencies.
