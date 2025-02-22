import cv2
import time
from ultralytics import YOLO

# Load the YOLOv11 model = ~110 MB model
model = YOLO("yolo_models/yolo11x.pt")

# Open the video file (uncomment the below lines if you want to use the samplevideo file)
# video_path = "videos/palace.mp4"
# cap = cv2.VideoCapture(video_path)

# Live video feed
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Get the video's original frame rate
fps = cap.get(cv2.CAP_PROP_FPS)
frame_duration = 1 / fps

while cap.isOpened():
    success, frame = cap.read()

    if success:
        start_time = time.time()

        results = model.track(frame, persist=True, conf=0.5, iou=0.5, tracker="bytetrack.yaml")
        annotated_frame = results[0].plot()
        cv2.putText(annotated_frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("YOLO Detection", annotated_frame)
        elapsed_time = time.time() - start_time
        wait_time = max(1, int((frame_duration - elapsed_time) * 1000))

        if cv2.waitKey(wait_time) & 0xFF == ord("q"):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()