This repository contains a custom object detection model trained to recognize various clothing items and accessories. Built using the YOLO11 architecture, the project includes the trained weights and Python scripts for running inference on both static images and real-time webcam feeds.

Dataset

The model was trained on a dataset sourced from Roboflow Universe.
Workspace: cutegander
Project: fashion-hkjfr

Recognized Classes (10):

sunglass, skirt, shorts, shoe, shirt, jacket, dress, hat, null, bag.

Training Details

Environment: Google Colab
Framework: Ultralytics
Epochs: 50
Data Ingestion: Direct download via Roboflow API

Repository Contents

best.pt: The fine-tuned YOLO11 model weights.
webcam_inference.py: Script for real-time object detection using a webcam.
image_inference.py: Script for running object detection on static images.
test1.jpg, test2.jpg: Two test images provided to quickly verify the model's functionality.

Prerequisites

Ensure you have Python installed on your system (tested on Python 3.13).
Note: The model was trained in a Linux environment (Colab) but is deployed locally on Windows. The provided scripts include a specific pathlib patch to automatically resolve PosixPath to WindowsPath compatibility errors during weight initialization.

Installation

To run the model locally, you only need to install the official ultralytics package, resolving and installing all required dependencies.

- Run the following command in your terminal or virtual environment:

pip install -r requirements.txt

Running the scripts:
- Running model on provided image:

python image_inference.py

- Real-time detection using webcam (pressing 'q' will close the window):

python webcam_inference.py

- Confidence threshold can be controlled using the following line:

results = model(img_path/frame, conf = 0.6)