import cv2
from ultralytics import YOLO

# Load the YOLO orientation model
model_orientation = YOLO("yolo_models/yolo11x-obb.pt")

# Open the webcam
cap = cv2.VideoCapture(0)

# Open the video file (uncomment the below lines if you want to use the video file)
# video_path = "videos/palace.mp4"
# cap = cv2.VideoCapture(video_path)

# Loop through the video frames
while cap.isOpened():
    success, frame = cap.read()

    if success:
        # Orientation
        results_orientation = model_orientation.predict(frame)
        annotated_frame_orientation = results_orientation[0].plot()

        # Display the annotated frame
        cv2.imshow("YOLO Orientation", annotated_frame_orientation)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()
