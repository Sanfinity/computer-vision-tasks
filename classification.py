import cv2
from ultralytics import YOLO

# Load the YOLO classification model
model_classification = YOLO("yolo_models/yolo11x-cls.pt")

# Open the webcam
cap = cv2.VideoCapture(0)

# Open the video file (uncomment the below lines if you want to use the video file)
# video_path = "videos/palace.mp4"
# cap = cv2.VideoCapture(video_path)

# Loop through the video frames
while cap.isOpened():
    success, frame = cap.read()

    if success:
        # Classification
        results_classification = model_classification.predict(frame, conf=0.5, iou=0.5)
        annotated_frame_classification = results_classification[0].plot()

        # Display the annotated frame
        cv2.imshow("YOLO Classification", annotated_frame_classification)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()
