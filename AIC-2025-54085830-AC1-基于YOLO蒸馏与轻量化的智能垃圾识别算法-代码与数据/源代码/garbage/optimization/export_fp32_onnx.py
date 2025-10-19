
from ultralytics import YOLO
m = YOLO("/root/autodl-tmp/SWS3009Assg/runs/detect/finetune_pruned/weights/best.pt")
m.export(format="onnx", imgsz=704, opset=13, simplify=True, dynamic=False)
