from ultralytics import YOLO
import os

def train_model():
    # Load pre-trained YOLOv8 small model
    model = YOLO("yolov8s.pt")

    # Start training
    results = model.train(
        data=os.path.abspath("traffic_light_dataset/data.yaml"),
        epochs=50,             # 50 epochs is a solid baseline for transfer learning
        imgsz=640,             # Standard resolution matching your Roboflow export
        batch=16,              # Adjust down to 8 if you hit GPU/CPU memory limits
        name="traffic_light_v1",
        patience=15,           # Early stopping if validation loss stops improving
        save=True,
        device="0" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"
    )

if __name__ == "__main__":
    train_model()
