import cv2
import numpy as np
from ultralytics import YOLO

def classify_light_color(crop_img):
    """
    Classifies Red or Green based on pixel counts. Yellow is entirely ignored.
    """
    if crop_img.size == 0:
        return "UNKNOWN"

    hsv = cv2.cvtColor(crop_img, cv2.COLOR_BGR2HSV)

    # Red spans both ends of the HSV hue wheel (0-10 and 160-180)
    lower_red1 = np.array([0, 70, 70])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 70, 70])
    upper_red2 = np.array([180, 255, 255])

    # Green range
    lower_green = np.array([40, 70, 70])
    upper_green = np.array([90, 255, 255])

    # Generate binary masks
    mask_r1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask_r2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask_red = cv2.bitwise_or(mask_r1, mask_r2)

    mask_green = cv2.inRange(hsv, lower_green, upper_green)

    # Count illuminated pixels
    red_count = cv2.countNonZero(mask_red)
    green_count = cv2.countNonZero(mask_green)

    min_pixel_threshold = 20

    # Determine winner
    if red_count > green_count and red_count > min_pixel_threshold:
        return "RED"
    elif green_count > red_count and green_count > min_pixel_threshold:
        return "GREEN"

    return "UNKNOWN"

def run_pipeline(video_path, model_path="best.pt", output_path="classified_output.mp4"):
    model = YOLO(model_path)
    cap = cv2.VideoCapture(video_path)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print(f"Processing video: {video_path}...")

    # Initialize the state machine
    previous_state = "UNKNOWN"

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Run YOLO detection
        results = model.predict(source=frame, conf=0.4, verbose=False)

        # Initialize the ballot box for this frame
        frame_votes = {"RED": 0, "GREEN": 0, "UNKNOWN": 0}

        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])

            # 1. Calculate the 15% margin to shrink the ROI
            box_width = x2 - x1
            box_height = y2 - y1
            margin_x = int(box_width * 0.15)
            margin_y = int(box_height * 0.15)

            # Apply the shrink
            sx1 = x1 + margin_x
            sy1 = y1 + margin_y
            sx2 = x2 - margin_x
            sy2 = y2 - margin_y

            # 2. Extract crop and run Red/Green classification
            if sx2 > sx1 and sy2 > sy1:
                crop = frame[sy1:sy2, sx1:sx2]
                status = classify_light_color(crop)
                frame_votes[status] += 1
            else:
                status = "UNKNOWN"

            # 3. Draw bounding boxes and labels
            color_bgr = (128, 128, 128)
            if status == "RED":
                color_bgr = (0, 0, 255)
            elif status == "GREEN":
                color_bgr = (0, 255, 0)

            # Draw the shrunk inner box (white) to visualize the cropped area
            cv2.rectangle(frame, (sx1, sy1), (sx2, sy2), (255, 255, 255), 1)
            # Draw the main YOLO bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color_bgr, 2)
            cv2.putText(frame, f"{status} {conf:.2f}", (x1, max(20, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_bgr, 2)

        # 4. Tally the votes for the entire frame
        current_state = "UNKNOWN"
        if frame_votes["GREEN"] > frame_votes["RED"]:
            current_state = "GREEN"
        elif frame_votes["RED"] > frame_votes["GREEN"]:
            current_state = "RED"
        else:
            # Tie-breaker or no active lights: Hold the previous known state
            if previous_state != "UNKNOWN":
                current_state = previous_state

        # 5. The Trigger: Catching the Red -> Green transition
        if previous_state == "RED" and current_state == "GREEN":
            # This is where the GPIO buzzer code will eventually fire!
            print(">>> ALERT: LIGHT TURNED GREEN! <<<")
            cv2.putText(frame, "!!! ALERT: GO !!!", (int(width/2) - 150, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 0), 5)

        # Display overall system state in the top left corner
        cv2.putText(frame, f"System State: {current_state}", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

        # Update the memory for the next loop
        previous_state = current_state

        out.write(frame)

    cap.release()
    out.release()
    print(f"Finished! Processed video saved to: {output_path}")

if __name__ == "__main__":
    run_pipeline("dashcam.ts")
