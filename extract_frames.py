import cv2
import os
import argparse

def extract_frames(video_path):
    # 1. Check if the video file exists
    if not os.path.exists(video_path):
        print(f"Error: The video file '{video_path}' does not exist.")
        return

    # 2. Create an output directory based on the video's name
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_dir = f"{video_name}_frames"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving frames to: {os.path.abspath(output_dir)}")

    # 3. Open the video file
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # 4. Get video properties (Frames Per Second)
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Detected Video FPS: {fps}")
    
    # CHANGED HERE: Multiply FPS by 2 to extract a frame every 2 seconds
    frame_interval = max(1, round(fps * 2)) 

    frame_count = 0
    saved_count = 0

    while True:
        success, frame = cap.read()
        
        # If the video ends, break the loop
        if not success:
            break

        # Check if the current frame lands exactly on the 2-second mark
        if frame_count % frame_interval == 0:
            # Format filename with leading zeros (e.g., frame_0001.jpg)
            filename = os.path.join(output_dir, f"frame_{saved_count:04d}.jpg")
            cv2.imwrite(filename, frame)
            saved_count += 1

        frame_count += 1

    # Clean up resources
    cap.release()
    print(f"Done! Successfully extracted {saved_count} frames (one every 2 seconds).")

if __name__ == "__main__":
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description="Extract one frame every 2 seconds from a video file.")
    parser.add_argument("video", type=str, help="Path to the video file (e.g., sample.mp4)")
    
    args = parser.parse_args()
    extract_frames(args.video)