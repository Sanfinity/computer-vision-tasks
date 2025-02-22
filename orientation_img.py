import cv2
from ultralytics import YOLO

# Load the YOLO orientation model
model_orientation = YOLO("yolo_models/yolo11x-obb.pt")  # Orientation model

# Load the image
image_path = "images/P0014.png"
# image_path = "images/P0015.png"
image = cv2.imread(image_path)

# Perform inference on the image
results_orientation = model_orientation.predict(image)
annotated_image_orientation = results_orientation[0].plot(line_width=1)

# Display the annotated image
cv2.imshow("YOLOv8 Orientation Results", annotated_image_orientation)
cv2.waitKey(0)  # Wait for a key press to close the window

# Close the display window
cv2.destroyAllWindows()
