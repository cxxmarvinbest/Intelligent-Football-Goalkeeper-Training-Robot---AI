import torch

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

from ultralytics import YOLO

if __name__ == '__main__':
    # 加载模型
    model = YOLO("yolov8n.pt")

    model.train(
        data="football.yaml",
        epochs=30,
        plots=False,
        batch=16,
        workers=0
    )