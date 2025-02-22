from ultralytics import YOLO
import cv2

# Load a YOLO model
model = YOLO("yolo_models/yolo11x-pose.pt")

# Open a connection to the webcam (0 is the default webcam)
cap = cv2.VideoCapture(0)

# Open the video file (uncomment the below lines if you want to use the video file)
# video_path = "videos/palace.mp4"
# cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image")
        break

    results = model(frame) 
    annotated_frame = results[0].plot()
    annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
    cv2.imshow("YOLO Pose Estimation", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close any OpenCV windows
cap.release()
cv2.destroyAllWindows()