import os
import shutil
import time
import subprocess
import sys

# Directory Configurations
RAW_DIR = "raw_videos"
STAGING_DIR = "staging_videos"
COMPLETED_DIR = "completed_videos"
EXTRACTED_DIR = "extracted_frames"
FILTERED_DIR = "traffic_lights_only"

BATCH_SIZE = 5
COOL_DOWN_SECONDS = 120  # 2 minutes cooling gap between stages

VIDEO_EXTENSIONS = ('.mp4', '.avi', '.mkv', '.mov', '.ts')

def setup_directories():
    """Ensure all required folders exist."""
    for folder in [RAW_DIR, STAGING_DIR, COMPLETED_DIR, EXTRACTED_DIR, FILTERED_DIR]:
        os.makedirs(folder, exist_ok=True)

def empty_directory(folder_path):
    """Delete all files inside a folder."""
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)

def cool_down_timer(seconds):
    """Visual countdown timer to let the CPU/GPU cool down."""
    print(f"\n[Thermal Cool-Down] Pausing for {seconds // 60} minutes to let the XPS desktop cool down...")
    for remaining in range(seconds, 0, -1):
        mins, secs = divmod(remaining, 60)
        timer = f"{mins:02d}:{secs:02d}"
        print(f"\rCooling down... {timer} remaining", end="", flush=True)
        time.sleep(1)
    print("\n[Cool-Down Complete] Resuming pipeline execution...\n")

def run_pipeline():
    setup_directories()

    batch_number = 1

    while True:
        # Find remaining videos in raw_videos
        all_videos = [f for f in os.listdir(RAW_DIR) if f.lower().endswith(VIDEO_EXTENSIONS)]

        if not all_videos:
            print("\n==================================================")
            print("🎉 All raw videos processed! Pipeline finished completely.")
            print("==================================================")
            break

        # Select up to BATCH_SIZE videos
        batch_videos = all_videos[:BATCH_SIZE]
        print(f"\n==================================================")
        print(f" Starting Batch {batch_number}: Processing {len(batch_videos)} video(s) ({len(all_videos)} remaining total)")
        print(f"==================================================")
        for vid in batch_videos:
            print(f"  • {vid}")

        # Move batch to staging_videos
        for vid in batch_videos:
            shutil.move(os.path.join(RAW_DIR, vid), os.path.join(STAGING_DIR, vid))

        try:
            # --- Stage 1: Curate & Extract Frames ---
            print("\n--- Stage 1: Extracting Curated Frames (curate_data.py) ---")
            env = os.environ.copy()
            env["VIDEO_DIR"] = STAGING_DIR
            subprocess.run([sys.executable, "curate_data.py"], env=env, check=True)

            # Mark videos as completed
            for vid in batch_videos:
                shutil.move(os.path.join(STAGING_DIR, vid), os.path.join(COMPLETED_DIR, vid))
            print(f"✓ Moved {len(batch_videos)} video(s) to '{COMPLETED_DIR}'.")

            # --- Intermission: Cool-Down Delay ---
            cool_down_timer(COOL_DOWN_SECONDS)

            # --- Stage 2: Filter Traffic Signals ---
            print("--- Stage 2: Filtering Traffic Lights (filter_traffic_lights.py) ---")
            subprocess.run([sys.executable, "filter_traffic_lights.py"], check=True)

            # --- Stage 3: Cleanup ---
            print("\n--- Cleaning Up Temporary Files ---")
            empty_directory(EXTRACTED_DIR)
            print(f"✓ Emptied '{EXTRACTED_DIR}' folder.")

            remaining_count = len([f for f in os.listdir(RAW_DIR) if f.lower().endswith(VIDEO_EXTENSIONS)])
            print(f"\nBatch {batch_number} complete! Remaining videos in '{RAW_DIR}': {remaining_count}")
            batch_number += 1

            # Brief pause between full batch cycles
            if remaining_count > 0:
                print("\nInitiating next batch cycle in 5 seconds...")
                time.sleep(5)

        except Exception as e:
            print(f"\n[ERROR] Pipeline failed on batch {batch_number}: {e}")
            # Recover staging videos back to raw_videos if process crashes
            for vid in os.listdir(STAGING_DIR):
                shutil.move(os.path.join(STAGING_DIR, vid), os.path.join(RAW_DIR, vid))
            break

if __name__ == "__main__":
    run_pipeline()
