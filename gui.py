import sys
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QPushButton,
    QFileDialog,
    QLabel,
    QVBoxLayout,
    QWidget,
    QRadioButton,
    QButtonGroup,
)
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtGui import QImage, QPixmap
import cv2
from tracker import Tracker
from ultralytics import YOLO
import pandas as pd
import time
import config


class VideoProcessor(QThread):
    update_status = pyqtSignal(str)
    update_frame = pyqtSignal(QImage)

    def __init__(self, video_source, is_realtime):
        super().__init__()
        self.video_source = video_source
        self.is_realtime = is_realtime
        self.running = True
        self.output_path = config.OUTPUT_VIDEO_PATH
        self.model = YOLO(config.YOLO_MODEL_PATH)

        with open(config.COCO_CLASSES_PATH, "r") as file:
            self.class_list = file.read().split("\n")

    def run(self):
        try:
            self.update_status.emit("Processing started...")
            self.process_video(self.video_source, self.is_realtime)
            self.update_status.emit("Processing completed.")
        except Exception as e:
            self.update_status.emit(f"Error: {e}")

    def process_video(self, video_source, is_realtime):
        cap = cv2.VideoCapture(video_source)

        tracker = Tracker()
        count = 0
        vh_down = {}
        counter = []
        vh_up = {}
        counter1 = []
        vh_dtime = {}
        vh_utime = {}
        cy1 = config.CY1
        cy2 = config.CY2
        offset = config.OFFSET

        if not is_realtime:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(
                self.output_path,
                fourcc,
                config.VIDEO_FPS,
                (config.VIDEO_WIDTH, config.VIDEO_HEIGHT),
            )

        while self.running:
            ret, frame = cap.read()
            if not ret:
                break

            count += 1
            if count % 3 != 0:
                continue

            frame = cv2.resize(frame, (config.VIDEO_WIDTH, config.VIDEO_HEIGHT))

            results = self.model.predict(frame)
            a = results[0].boxes.data
            px = pd.DataFrame(a).astype("float")

            vehicles = []
            for index, row in px.iterrows():
                x1, y1, x2, y2, _, d = map(int, row)
                c = self.class_list[d]

                if c in ["car", "truck", "bus"]:
                    vehicles.append([x1, y1, x2, y2])

            bbox_id = tracker.update(vehicles)

            for bbox in bbox_id:
                x3, y3, x4, y4, id = bbox
                cx = (x3 + x4) // 2
                cy = (y3 + y4) // 2

                if cy1 - offset < cy < cy1 + offset:
                    vh_down[id] = cy
                    vh_dtime[id] = time.time()

                if cy2 - offset < cy < cy2 + offset:
                    vh_up[id] = cy
                    vh_utime[id] = time.time()

                cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 0, 255), 1)
                if id in vh_down:
                    elapsed_time = time.time() - vh_dtime[id]
                    self.display_speed(frame, cx, cy, id, counter, elapsed_time, x4, y4)
                if id in vh_up:
                    elapsed_time2 = time.time() - vh_utime[id]
                    self.display_speed(
                        frame, cx, cy, id, counter1, elapsed_time2, x4, y4
                    )

            cv2.line(frame, (267, cy1), (829, cy1), (255, 255, 255), 1)
            cv2.putText(
                frame,
                "1line",
                (274, 318),
                cv2.FONT_HERSHEY_COMPLEX,
                0.8,
                (0, 255, 255),
                2,
            )

            cv2.line(frame, (167, cy2), (932, cy2), (255, 255, 255), 1)
            cv2.putText(
                frame,
                "2line",
                (181, 363),
                cv2.FONT_HERSHEY_COMPLEX,
                0.8,
                (0, 255, 255),
                2,
            )

            cv2.putText(
                frame,
                "GoingDown: " + str(len(counter)),
                (60, 40),
                cv2.FONT_HERSHEY_COMPLEX,
                0.7,
                (255, 0, 0),
                2,
            )
            cv2.putText(
                frame,
                "GoingUp: " + str(len(counter1)),
                (60, 130),
                cv2.FONT_HERSHEY_COMPLEX,
                0.7,
                (255, 0, 0),
                2,
            )

            if not is_realtime:
                out.write(frame)

            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qt_image = QImage(
                rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888
            )
            self.update_frame.emit(qt_image)

            if cv2.waitKey(1) & 0xFF == 27:
                break

        cap.release()
        if not is_realtime:
            out.release()

    def stop(self):
        self.running = False

    def display_speed(
        self, frame, cx, cy, vehicle_id, counter_list, elapsed_time, x, y
    ):
        cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
        cv2.putText(
            frame,
            str(vehicle_id),
            (cx, cy),
            cv2.FONT_HERSHEY_COMPLEX,
            0.8,
            (0, 255, 255),
            2,
        )

        if vehicle_id not in counter_list:
            counter_list.append(vehicle_id)

        distance = config.DISTANCE

        if elapsed_time > 0:
            speed_ms = distance / elapsed_time
            speed_km = speed_ms * 3.6
            cv2.putText(
                frame,
                str(int(speed_km)) + " Km/h",
                (x, y),
                cv2.FONT_HERSHEY_COMPLEX,
                0.8,
                (0, 255, 255),
                2,
            )
        else:
            cv2.putText(
                frame,
                "Speed N/A",
                (x, y),
                cv2.FONT_HERSHEY_COMPLEX,
                0.8,
                (0, 255, 255),
                2,
            )


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Vehicle Speed Detection")
        self.setGeometry(100, 100, 800, 600)
        self.initUI()
        self.video_source = None
        self.processor = None

    def initUI(self):
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        self.video_label = QLabel(self)
        self.layout.addWidget(self.video_label)

        self.radio_button_group = QButtonGroup(self)

        self.file_radio = QRadioButton("Process Video File")
        self.file_radio.setChecked(True)
        self.radio_button_group.addButton(self.file_radio)
        self.layout.addWidget(self.file_radio)

        self.realtime_radio = QRadioButton("Real-time Detection")
        self.radio_button_group.addButton(self.realtime_radio)
        self.realtime_radio.toggled.connect(self.on_realtime_toggle)
        self.layout.addWidget(self.realtime_radio)

        self.select_button = QPushButton("Select Video", self)
        self.select_button.clicked.connect(self.select_video)
        self.layout.addWidget(self.select_button)

        self.start_button = QPushButton("Start Processing", self)
        self.start_button.clicked.connect(self.start_processing)
        self.layout.addWidget(self.start_button)

        self.stop_button = QPushButton("Stop Processing", self)
        self.stop_button.clicked.connect(self.stop_processing)
        self.stop_button.setEnabled(False)
        self.layout.addWidget(self.stop_button)

        self.status_label = QLabel(self)
        self.layout.addWidget(self.status_label)

    def on_realtime_toggle(self, checked):
        if checked:
            self.video_source = 0  # Use 0 for the default webcam
            self.start_processing()
            self.select_button.setEnabled(False)
            self.start_button.setEnabled(False)
        else:
            self.select_button.setEnabled(True)
            self.start_button.setEnabled(True)

    def select_video(self):
        if self.file_radio.isChecked():
            video_source, _ = QFileDialog.getOpenFileName(self, "Select Video File")
            if video_source:
                self.video_source = video_source
                self.start_button.setEnabled(True)

    def start_processing(self):
        if self.video_source is not None:
            self.processor = VideoProcessor(
                self.video_source, self.realtime_radio.isChecked()
            )
            self.processor.update_status.connect(self.update_status)
            self.processor.update_frame.connect(self.update_frame)
            self.processor.start()
            self.start_button.setEnabled(False)
            self.stop_button.setEnabled(True)

    def stop_processing(self):
        if self.processor:
            self.processor.stop()
            self.processor.wait()
            self.start_button.setEnabled(True)
            self.stop_button.setEnabled(False)
            self.status_label.setText("Processing stopped.")

    def update_status(self, message):
        self.status_label.setText(message)

    def update_frame(self, image):
        self.video_label.setPixmap(QPixmap.fromImage(image))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    mainWindow = MainWindow()
    mainWindow.show()
    sys.exit(app.exec_())
