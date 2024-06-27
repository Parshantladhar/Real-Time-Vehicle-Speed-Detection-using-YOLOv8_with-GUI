import os

# Get the directory of the current file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Path to the YOLO model
YOLO_MODEL_PATH = os.path.join(BASE_DIR, "yolov8s.pt")

# Path to the COCO class labels file
COCO_CLASSES_PATH = os.path.join(BASE_DIR, "coco.txt")

# Output video path
OUTPUT_VIDEO_PATH = os.path.join(BASE_DIR, "output.mp4")

# Constants for vehicle tracking
CY1 = 322
CY2 = 368
OFFSET = 6

# Video capture settings
VIDEO_WIDTH = 1020
VIDEO_HEIGHT = 500
VIDEO_FPS = 20.0

# Other settings
DISTANCE = 10  # Distance in meters for speed calculation
