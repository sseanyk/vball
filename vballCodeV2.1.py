import cv2
import numpy as np
from moviepy import VideoFileClip
import os

def analyze_airspace_for_ball(video_path, min_rally_sec=3.0, max_gap_sec=2.0, buffer_sec=1.5, start_sec=0, min_interactions=3, min_arc_gap_sec=0.5):
    """
    Detects rallies by looking for ball-sized movement in the upper airspace of the court.
    Includes interaction counting to filter out false positives (e.g., a bird flying by).
    """
    cap = cv2.VideoCapture(video_path)
    if start_sec > 0:
        cap.set(cv2.CAP_PROP_POS_MSEC, start_sec * 1000)
        print(f"Skipping ahead to {start_sec} seconds...")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    ret, first_frame = cap.read()
    if not ret:
        print("Error: Could not open video file.")
        return []

    # 1. Select ONLY the air above the players
    print("\n--- INSTRUCTIONS ---")
    print("1. Drag a box around the AIRSPACE ABOVE THE NET.")
    print("   (Do NOT include the players' bodies. We only want the area where the ball flies!)")
    print("2. Press ENTER or SPACE when done.")
    roi = cv2.selectROI("Select Airspace", first_frame, showCrosshair=True)
    cv2.destroyAllWindows()
    
    x, y, w, h = [int(v) for v in roi]
    if w == 0 or h == 0:
        print("No region selected. Exiting.")
        return []

    # 2. Background Subtractor tuned for small, fast objects
    back_sub = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=16, detectShadows=False)
    
    active_seconds = []
    frame_idx = 0

    print("\nAnalyzing ball movement in airspace...")
    print("Press 'q' on the video window to stop early.")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % 2 == 0:
            # Crop to the airspace
            air_crop = frame[y:y+h, x:x+w]
            
            # Get motion mask
            fg_mask = back_sub.apply(air_crop)
            
            # Clean up tiny digital noise (static)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
            
            # Find distinct moving objects (contours)
            contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            ball_detected = False
            for cnt in contours:
                area = cv2.contourArea(cnt)

                # 1. PRINT THE SIZES TO THE TERMINAL
            #    if area > 150:  # Ignore microscopic dust
            #        print(f"Time: {frame_idx/fps:.1f}s | Moving Object Size: {area:.1f} pixels")     
         
           
                # 2. FILTER BY SIZE
                if 200 < area < 800: 
                    bx, by, bw, bh = cv2.boundingRect(cnt)
                    
                    # Calculate aspect ratio (width / height)
                    aspect_ratio = float(bw) / bh
                    
                    # A ball usually has an aspect ratio between 0.5 and 2.0.
                    # Heads/arms connected to bodies often create extreme ratios (< 0.4 or > 2.5).
                    if aspect_ratio < 0.4 or aspect_ratio > 2.5:
                        continue
                    
                    ball_detected = True
                    
            if ball_detected:
                # Add start_sec to ensure the clip timestamp aligns with the original video
                active_seconds.append(start_sec + (frame_idx / fps))

            # FOR TESTING 3. SHOW THE LIVE VIDEO WINDOWS
        #    cv2.imshow("Airspace Video (Press 'q' to quit)", air_crop)
        #    cv2.imshow("Ball Motion Mask", fg_mask)
                
        #    if cv2.waitKey(1) & 0xFF == ord('q'):
        #        break

        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()

    # 4. Group continuous timestamps into rallies (Interaction-Based)
    if not active_seconds:
        print("No ball movement detected. Try lowering the minimum area size.")
        return []

    rallies = []
    rally_start = active_seconds[0]
    prev_sec = active_seconds[0]
    
    # Track how many distinct arcs/interactions happen in this rally
    interaction_count = 1 

    for sec in active_seconds[1:]:
        gap = sec - prev_sec
        
        # If the gap is larger than max_gap_sec, the rally is over
        if gap > max_gap_sec:
            rally_end = prev_sec
            
            # Check BOTH time and interaction thresholds!
            if (rally_end - rally_start) >= min_rally_sec and interaction_count >= min_interactions:
                rallies.append((max(0, rally_start - buffer_sec), rally_end + buffer_sec))
                
            # Reset for the next potential rally
            rally_start = sec
            interaction_count = 1
            
        # If the gap is smaller than max_gap_sec, but large enough to mean 
        # the ball left the airspace (e.g. > 0.5 seconds), it's a new interaction!
        elif gap > min_arc_gap_sec:
            interaction_count += 1
            
        prev_sec = sec

    # Don't forget to check the very last rally in the video
    if (prev_sec - rally_start) >= min_rally_sec and interaction_count >= min_interactions:
        rallies.append((max(0, rally_start - buffer_sec), prev_sec + buffer_sec))

    return rallies


def export_clips(input_video, rallies, output_folder="V2.1court_rallies"):
    base_name = os.path.splitext(os.path.basename(input_video))[0]
    output_folder = os.path.join(output_folder, base_name)
    os.makedirs(output_folder, exist_ok=True)
    clip = VideoFileClip(input_video)

    print(f"\nExporting {len(rallies)} rallies to '{output_folder}/'...")
    for i, (start, end) in enumerate(rallies):
        end_time = min(end, clip.duration)
        subclip = clip.subclipped(start, end_time) 
        output_file = os.path.join(output_folder, f"{base_name}_rally_{i+1:02d}_({start:.1f}s-{end_time:.1f}s).mp4")
        subclip.write_videofile(output_file, codec="libx264", audio_codec="aac")
        print(f"Saved: {output_file} ({start:.1f}s - {end_time:.1f}s)")

    clip.close()


if __name__ == "__main__":
    video_path = "0726_game_9.mp4"
    
    detected_rallies = analyze_airspace_for_ball(
        video_path, 
        min_rally_sec=3.0,   
        max_gap_sec=3.5,     # Allowed time the ball is out of the "airspace" (e.g. during a dig or pass)
        buffer_sec=2.5,      # Grab 2.5 seconds before and after the ball enters the airspace
        start_sec=0,
        min_interactions=3,  # NEW: Must see at least 3 distinct arcs
        min_arc_gap_sec=0.5  # NEW: Ball must vanish for 0.5s to count as a new arc
    )
    
    if detected_rallies:
        print(f"\nFound {len(detected_rallies)} rallies!")
        export_clips(video_path, detected_rallies)