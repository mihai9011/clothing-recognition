import cv2
from ultralytics import YOLO
import pathlib


pathlib.PosixPath = pathlib.WindowsPath
try:
    import pathlib._local
    pathlib._local.PosixPath = pathlib._local.WindowsPath
except ImportError:
    pass

model = YOLO('best.pt')

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    
    if not ret:
        break

    results = model(frame, conf = 0.4)

    # extracts and renders image
    annotated_frame = results[0].plot()
    
    cv2.imshow('YOLO11 clothing recognition', annotated_frame)

    # press q to close window
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()