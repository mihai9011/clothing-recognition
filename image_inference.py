from ultralytics import YOLO
import pathlib

pathlib.PosixPath = pathlib.WindowsPath
try:
    import pathlib._local
    pathlib._local.PosixPath = pathlib._local.WindowsPath
except ImportError:
    pass

model = YOLO('best.pt')

img_path = 'test1.jpg'

results = model(img_path, conf = 0.6)

results[0].show()

