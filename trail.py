# This is just a trail file which you can use to tweak your code further
import cv2
from ultralytics import YOLO

# Load the YOLOv11 model
# model_detection = YOLO("yolo_models/yolo11x.pt")  # Detection model
# model_classification = YOLO("yolo_models/yolo11x-cls.pt")  # Classification model
# model_orientation = YOLO("yolo_models/yolo11x-obb.pt")  # Orientation model
# model_segmentation = YOLO("yolo_models/yolo11x-seg.pt")  # Segmentation model
# model_pose = YOLO("yolo_models/yolo11x-pose.pt")  # Pose estimation model
# For more on models refer - https://docs.ultralytics.com/models/
model = YOLO("yolo_models/yolo11x-pose.pt")

# Path to the input image
input_image_path = 'sample8.png'
# Path to save the output image
output_image_path = 'sample2out.png'

# Load the input image
image = cv2.imread(input_image_path)

# Perform object detection and tracking
results = model(image)

# Annotate the image with the tracking results
annotated_image = results[0].plot(line_width=1)

# Save the annotated image
cv2.imwrite(output_image_path, annotated_image)

# Display the annotated image
cv2.imshow("YOLOv8 Tracking", annotated_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
