import cv2
import time
from tracker import Tracker
import config
import pandas as pd
from ultralytics import YOLO


def load_model(model_path):
    return YOLO(model_path)


def load_class_list(class_list_path):
    with open(class_list_path, "r") as file:
        return file.read().split("\n")


def calculate_speed(elapsed_time, distance):
    if elapsed_time > 0:
        speed_ms = distance / elapsed_time
        speed_km = speed_ms * 3.6
        return speed_km
    return 0


def display_speed(frame, cx, cy, vehicle_id, counter_list, elapsed_time, x, y):
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

    distance = 10

    speed_km = calculate_speed(elapsed_time, distance)
    if speed_km > 0:
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
            frame, "Speed N/A", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2
        )


def process_frame(frame, model, class_list, tracker, counters, times):
    cy1 = 322
    cy2 = 368
    offset = 6

    results = model.predict(frame)
    a = results[0].boxes.data
    px = pd.DataFrame(a).astype("float")

    vehicles = []

    for index, row in px.iterrows():
        x1, y1, x2, y2, _, d = map(int, row)
        c = class_list[d]

        if c in ["car", "truck", "bus"]:
            vehicles.append([x1, y1, x2, y2])

    bbox_id = tracker.update(vehicles)

    for bbox in bbox_id:
        x3, y3, x4, y4, id = bbox
        cx = (x3 + x4) // 2
        cy = (y3 + y4) // 2

        if cy1 - offset < cy < cy1 + offset:
            times[id] = time.time()

        if cy2 - offset < cy < cy2 + offset and id in times:
            elapsed_time = time.time() - times[id]
            display_speed(frame, cx, cy, id, counters["down"], elapsed_time, x4, y4)
            del times[id]

        cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 0, 255), 1)

    cv2.line(frame, (267, cy1), (829, cy1), (255, 255, 255), 1)
    cv2.putText(
        frame, "1line", (274, 318), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2
    )

    cv2.line(frame, (167, cy2), (932, cy2), (255, 255, 255), 1)
    cv2.putText(
        frame, "2line", (181, 363), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2
    )

    cv2.putText(
        frame,
        "GoingDown: " + str(len(counters["down"])),
        (60, 40),
        cv2.FONT_HERSHEY_COMPLEX,
        0.7,
        (255, 0, 0),
        2,
    )
    cv2.putText(
        frame,
        "GoingUp: " + str(len(counters["up"])),
        (60, 130),
        cv2.FONT_HERSHEY_COMPLEX,
        0.7,
        (255, 0, 0),
        2,
    )

    return frame


def main(video_path=config.VIDEO_PATH, output_path=config.OUTPUT_VIDEO_PATH):
    model = load_model(config.MODEL_PATH)
    class_list = load_class_list(config.COCO_CLASSES_PATH)
    cap = cv2.VideoCapture(video_path)
    tracker = Tracker()

    out = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        20.0,
        (config.FRAME_WIDTH, config.FRAME_HEIGHT),
    )

    counters = {"down": [], "up": []}
    times = {}

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (config.FRAME_WIDTH, config.FRAME_HEIGHT))
        frame = process_frame(frame, model, class_list, tracker, counters, times)

        out.write(frame)
        cv2.imshow("RGB", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
