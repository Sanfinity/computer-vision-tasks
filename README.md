# Computer Vision Tasks  

This repository contains various computer vision tasks using the **Ultralytics YOLO** model for object detection, classification, segmentation, tracking, and pose estimation.  

## 📂 Folder Structure  
```
──📦 computer-vision-tasks
   ├── 📂 images                # Stores sample images
   ├── 📂 videos                # Stores sample videos
   ├── 📂 yolo_models           # Contains YOLO model weights
   ├── 📜 classification.py     # Classification using YOLO
   ├── 📜 detection.py          # Object detection using YOLO
   ├── 📜 orientation.py        # Orientation detection using YOLO
   ├── 📜 pose_estimation.py    # Pose estimation using YOLO
   ├── 📜 segmentation.py       # Segmentation using YOLO
   ├── 📜 tracking.py           # Object tracking with YOLO
   ├── 📜 requirements.txt      # Required dependencies
   └── 📜 README.md             # Project documentation
```

## ⚙️ Installation  

1. Clone this repository:  
   ```bash
   git clone https://github.com/your-repo/computer-vision-tasks.git
   cd computer-vision-tasks
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. (Optional) If you want to use GPU acceleration, install PyTorch (check [here](https://pytorch.org/) for latest):
    ```bash
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    ```
    (Make sure to install the correct version based on your CUDA version.)

## 📥 Download Models & Sample Data
To get the YOLO models, sample images, and videos, download the [v1 release](https://github.com/Sanfinity/computer-vision-tasks/releases/download/v1/data.zip) and extract them into the project folder.

## 🚀 Usage
All scripts use the webcam by default. If you want to use a sample video, uncomment the relevant lines in the script.

- To run a script, for example, detection.py, use:
    ```bash
    python detection.py
    ```
    Modify the script as needed to process images or videos instead of the webcam.

---
###### Made with ❤️ by [Sanfinity](https://github.com/Sanfinity/)
