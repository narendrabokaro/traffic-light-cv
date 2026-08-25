# Smart Traffic Light Assistant (Edge AI)

A lightweight, hybrid computer vision pipeline designed to detect **Red-to-Green** traffic light transitions and trigger hardware alerts. 

This project bridges deep learning and traditional computer vision to create an efficient driver-assistance tool. It is architected specifically for low-power edge devices, utilizing a neural processing unit (NPU) for object detection and a CPU for fast, deterministic state tracking.

## System Architecture

The pipeline uses a "division of labor" approach to maximize frames-per-second (FPS) on edge hardware:

1. **YOLOv8 (NPU):** Scans the dashcam frame and returns bounding boxes for all visible traffic light housings.
2. **OpenCV (CPU):** Processes the cropped Regions of Interest (ROIs). It converts the BGR image to HSV and applies strict binary color masks for **Red** and **Green** (ignoring Yellow entirely to avoid false positives from painted backplates).
3. **State Machine:** Tallies an "All-Box Majority Vote" across the multi-lane intersection. When it registers a definitive **Red to Green** transition, it triggers an alert flag.
4. **Cooldown Logic:** Upon a successful transition, the system enters a 5-second sleep state, bypassing NPU and CPU processing while the vehicle clears the intersection, before wiping its memory back to an `UNKNOWN` state.

## Hardware & Tech Stack

*   **Development OS:** Arch Linux
*   **Target Edge Hardware:** Orange Pi 5 Pro
*   **Computer Vision:** OpenCV, NumPy
*   **Deep Learning:** YOLOv8 (PyTorch)
*   **Model Pipeline:** `best.pt` -> ONNX -> RKNN (INT8 Quantization planned)
*   **Dataset Annotation:** Roboflow

## Current Status

The system is currently running as a unified desktop prototype (`traffic_assist_core.py`) for video file validation. 

**Latest Optimizations:**
*   Eliminated yellow HSV masking to reduce CPU load by 33%.
*   Removed ROI inner-cropping to improve accuracy on horizontal mast-arm signals.
*   Implemented a 5-second inference cooldown post-transition.

## Installation & Usage (Desktop Testing)

1. Clone the repository
```bash
git clone https://github.com/narendrabokaro/traffic_light_cv.git
cd traffic_light_cv
```

3. Install dependencies
```bash
pip install opencv-python numpy ultralytics
```

5. Run the pipeline
Place your YOLOv8 weights (best.pt) and your dashcam test video (dashcam.mp4) in the root directory, then execute:
```bash
python traffic_assist_core.py
```

The script will output a classified_output.mp4 file with bounding boxes, color classifications, and system state overlays.
Development Roadmap

    [x] Train baseline YOLOv8 model on custom traffic light dataset.

    [x] Implement hybrid YOLO + OpenCV pipeline.

    [x] Develop Red/Green state machine and cooldown logic.

    [ ] [WIP] Branch into refactor/modular-architecture to split logic into vision_engine.py and hardware_controller.py.

    [ ] Export ONNX model and compile to .rknn with INT8 quantization.

    [ ] Deploy to Orange Pi 5 Pro.

    [ ] Integrate GPIO output for physical piezoelectric buzzer alerts.
