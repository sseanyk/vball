import cv2
import numpy as np
from moviepy import VideoFileClip

def analyze_rally_motion(video_path, sensitivity=0.03, min_rally_sec=3.0, max_gap_sec=2.5, buffer_sec=1.5):
    """
    Detects active rallies by measuring motion intensity within a user-defined Court ROI.
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    ret, first_frame = cap.read()
    if not ret:
        print("Error: Could not open video file.")
        return []

    # 1. Let the user draw a box around the active court area
    print("\n--- INSTRUCTIONS ---")
    print("1. Drag a box around the MAIN COURT PLAY AREA (where players hit/jump).")
    print("2. Press ENTER or SPACE when done.")
    roi = cv2.selectROI("Select Court Area", first_frame, showCrosshair=True)
    cv2.destroyAllWindows()
    
    x, y, w, h = [int(v) for v in roi]
    court_area_pixels = w * h
    if court_area_pixels == 0:
        print("No region selected. Exiting.")
        return []

    # 2. Background Subtractor optimized for fixed cameras
    back_sub = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=40, detectShadows=False)
    
    active_seconds = []
    frame_idx = 0

    print("\nAnalyzing video motion...")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Process every 2nd frame to speed up calculation



        if frame_idx % 2 == 0:
            # Crop to court ROI
            court_crop = frame[y:y+h, x:x+w]
            
            # Get motion mask
            fg_mask = back_sub.apply(court_crop)
            
            # Count moving (white) pixels in court area
            moving_pixels = cv2.countNonZero(fg_mask)
            motion_ratio = moving_pixels / float(court_area_pixels)
            
            # COUNTING 
            print(f"Time: {frame_idx/fps:.1f}s | Motion Ratio: {motion_ratio:.4f}")

            # ---> ADD THESE 4 LINES TO SEE THE VIDEO <---
            cv2.imshow("Live Video", court_crop)
            cv2.imshow("Computer Vision Mask", fg_mask)
            if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit early
                break
            # --------------------------------------------


            # Flag frame if motion passes sensitivity threshold
            if motion_ratio > sensitivity:
                active_seconds.append(frame_idx / fps)

        frame_idx += 1

    cap.release()

    # 3. Group continuous active timestamps into rallies
    if not active_seconds:
        print("No significant activity detected. Try lowering the 'sensitivity' value.")
        return []

    rallies = []
    rally_start = active_seconds[0]
    prev_sec = active_seconds[0]

    for sec in active_seconds[1:]:
        # If gap between motion spurts is larger than max_gap_sec, mark end of rally
        if sec - prev_sec > max_gap_sec:
            rally_end = prev_sec
            if (rally_end - rally_start) >= min_rally_sec:
                rallies.append((max(0, rally_start - buffer_sec), rally_end + buffer_sec))
            rally_start = sec
        prev_sec = sec

    # Add final clip
    if (prev_sec - rally_start) >= min_rally_sec:
        rallies.append((max(0, rally_start - buffer_sec), prev_sec + buffer_sec))

    return rallies

def export_clips(input_video, rallies, output_folder="court_rallies"):
    """
    Exports detected rally timestamps into separate MP4 files.
    """
    import os
    os.makedirs(output_folder, exist_ok=True)
    
    clip = VideoFileClip(input_video)
    
    print(f"\nExporting {len(rallies)} rallies to '{output_folder}/'...")
    for i, (start, end) in enumerate(rallies):
        end_time = min(end, clip.duration)
        subclip = clip.subclipped(start, end_time)
        
        output_file = os.path.join(output_folder, f"rally_{i+1:02d}.mp4")
        subclip.write_videofile(output_file, codec="libx264", audio_codec="aac")
        print(f"Saved: {output_file} ({start:.1f}s - {end_time:.1f}s)")

    clip.close()

if __name__ == "__main__":
    video_path = "0727.mp4"
    
    # Analyze video using Court ROI motion analysis
    detected_rallies = analyze_rally_motion(
        video_path, 
        sensitivity=0.11,  # Adjust lower (e.g. 0.015) if missing rallies; higher if catching standing players
        min_rally_sec=3.0,   # Ignore motion bursts shorter than 3 seconds
        max_gap_sec=3.5,     # Allow up to 2.5s of low motion during high sets/passes
        buffer_sec=2.5       # Extra seconds before service and after ball drop
    )
    
    if detected_rallies:
        print(f"\nFound {len(detected_rallies)} rallies!")
        export_clips(video_path, detected_rallies)