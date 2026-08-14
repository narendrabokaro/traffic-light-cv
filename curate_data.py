import cv2
import os
import numpy as np
from PIL import Image
import imagehash

# Configuration
VIDEO_DIR = os.getenv("VIDEO_DIR", "raw_videos")
OUTPUT_DIR = "extracted_frames"
FRAME_SKIP = 30           # Extract 1 frame every second on 30 FPS
BLUR_THRESHOLD = 100.0    # Higher = stricter blur filter. Adjust based on your dashcam quality.
HASH_DIFFERENCE = 10       # How different images must be to be kept (prevents duplicates - more strict rule, try 8 if this is too much)

def variance_of_laplacian(image):
    # Computes the Laplacian of the image and returns the variance (blur metric)
    return cv2.Laplacian(image, cv2.CV_64F).var()

def process_videos():
    seen_hashes = []
    saved_count = 0

    for video_file in os.listdir(VIDEO_DIR):
        video_path = os.path.join(VIDEO_DIR, video_file)
        if not os.path.isfile(video_path):
            continue

        print(f"Processing: {video_file}")
        cap = cv2.VideoCapture(video_path)
        frame_id = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Skip frames to reduce processing time
            if frame_id % FRAME_SKIP != 0:
                frame_id += 1
                continue

            # 1. Blur Detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            fm = variance_of_laplacian(gray)

            if fm < BLUR_THRESHOLD:
                frame_id += 1
                continue # Skip blurry frame

            # 2. Duplicate Detection (Perceptual Hashing)
            pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            current_hash = imagehash.phash(pil_img)

            is_duplicate = False
            for h in seen_hashes:
                if current_hash - h < HASH_DIFFERENCE:
                    is_duplicate = True
                    break

            if is_duplicate:
                frame_id += 1
                continue # Skip duplicate frame

            # 3. Save Valid Frame
            seen_hashes.append(current_hash)
            out_name = f"frame_{video_file}_{frame_id}.jpg"
            cv2.imwrite(os.path.join(OUTPUT_DIR, out_name), frame)
            saved_count += 1

            frame_id += 1

        cap.release()

    print(f"Extraction complete! Saved {saved_count} highly curated frames.")

if __name__ == "__main__":
    process_videos()
