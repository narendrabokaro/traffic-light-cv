import os
import shutil
from ultralytics import YOLO

# Folders
INPUT_DIR = "extracted_frames"
OUTPUT_DIR = "traffic_lights_only"

# Create output folder if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load lightweight pre-trained COCO YOLOv8 model (downloads automatically on first run ~6MB)
model = YOLO("yolov8s.pt")

# Confidence threshold for detecting a traffic light (0.25 = 25% certainty)
CONFIDENCE_THRESHOLD = 0.20

def filter_images():
    images = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    kept_count = 0

    print(f"Scanning {len(images)} images for traffic lights...")

    for img_name in images:
        img_path = os.path.join(INPUT_DIR, img_name)

        # Run inference
        results = model(img_path, imgsz=800, verbose=False)[0]

        has_traffic_light = False

        # Check detected objects
        for box in results.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            class_name = model.names[cls_id]

            # COCO class for traffic light
            if class_name == "traffic light" and conf >= CONFIDENCE_THRESHOLD:
                has_traffic_light = True
                break

        # Move image to output directory if a traffic light is detected
        if has_traffic_light:
            shutil.copy(img_path, os.path.join(OUTPUT_DIR, img_name))
            kept_count += 1

    print(f"\nFiltering complete!")
    print(f"Total scanned: {len(images)}")
    print(f"Saved to '{OUTPUT_DIR}': {kept_count} images")

if __name__ == "__main__":
    filter_images()
