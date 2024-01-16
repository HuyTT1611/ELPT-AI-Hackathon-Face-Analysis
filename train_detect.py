import torch
from ultralytics import YOLO

def main():
    print(torch.cuda.is_available())
    model = YOLO('yolov8n.pt')
    model.train(data='./configs/detect.yaml', epochs=200, batch=16, imgsz=640, plots=True, device=[0], workers=2)

if __name__ == '__main__':
    main()