import cv2
from ultralytics import YOLO

# Load the YOLO models
model_classification = YOLO("yolo_models/yolo11x-cls.pt")  # Classification model
model_orientation = YOLO("yolo_models/yolo11x-obb.pt")  # Orientation model
model_segmentation = YOLO("yolo_models/yolo11x-seg.pt")  # Segmentation model
model_pose = YOLO("yolo_models/yolo11x-pose.pt")  # Pose estimation model

# Open the webcam
cap = cv2.VideoCapture(0)

# Loop through the video frames
while cap.isOpened():
    success, frame = cap.read()

    if success:
        # Classification
        results_classification = model_classification.predict(frame)
        annotated_frame_classification = results_classification[0].plot(line_width=1)

        # Orientation
        results_orientation = model_orientation.predict(frame)
        annotated_frame_orientation = results_orientation[0].plot(line_width=1)

        # Segmentation
        results_segmentation = model_segmentation.predict(frame)
        annotated_frame_segmentation = results_segmentation[0].plot(line_width=1)

        # Pose estimation
        results_pose = model_pose.predict(frame)
        annotated_frame_pose = results_pose[0].plot(line_width=1)

        # Resize frames to ensure they have the same size
        height, width = annotated_frame_classification.shape[:2]
        annotated_frame_orientation = cv2.resize(annotated_frame_orientation, (width, height))
        annotated_frame_segmentation = cv2.resize(annotated_frame_segmentation, (width, height))
        annotated_frame_pose = cv2.resize(annotated_frame_pose, (width, height))

        # Combine the results into a 2x2 grid
        top_row = cv2.hconcat([annotated_frame_classification, annotated_frame_orientation])
        bottom_row = cv2.hconcat([annotated_frame_segmentation, annotated_frame_pose])
        combined_frame = cv2.vconcat([top_row, bottom_row])

        # Display the combined frame
        cv2.imshow("YOLO Combined Results", combined_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()
