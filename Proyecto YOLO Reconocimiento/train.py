from ultralytics import YOLO

# Cargar modelo base
model = YOLO('yolov8n.pt')

# Entrenar con tus datos
model.train(
    data='datasets/mi_dataset/data.yaml',
    epochs=49,
    imgsz=640
)