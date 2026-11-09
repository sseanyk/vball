import cv2
import os
from moviepy import VideoFileClip
from ultralytics import YOLO
import shutil

def analyze_airspace_with_custom_yolo(video_path, start_sec=0.0, model_path="volleyball_yolo.pt", min_rally_sec=9, max_gap_sec=3.5, buffer_sec=2.5):
    """
    Detects rallies using a custom-trained YOLO AI model downloaded from an existing repository.
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # --- NEW: Fast-forward to the desired start time (in milliseconds) ---
    if start_sec > 0:
        cap.set(cv2.CAP_PROP_POS_MSEC, start_sec * 1000)
    
    ret, first_frame = cap.read()
    if not ret:
        print("Error: Could not open video file.")
        return []

    print("\n--- INSTRUCTIONS ---")
    print("1. Drag a box around the AIRSPACE ABOVE THE NET.")
    print("2. Press ENTER or SPACE when done.")
    roi = cv2.selectROI("Select Airspace", first_frame, showCrosshair=True)
    cv2.destroyAllWindows()
    
    x, y, w, h = [int(v) for v in roi]
    if w == 0 or h == 0:
        return []

    # 1. LOAD THE CUSTOM REPOSITORY MODEL
    if not os.path.exists(model_path):
        print(f"Error: Could not find '{model_path}'. Did you download it from GitHub and put it in this folder?")
        return []
        
    print(f"\nLoading Custom AI Model: {model_path}...")
    model = YOLO(model_path)
    
    active_seconds = []
    frame_idx = 0

    print("\nAnalyzing ball movement...")
    
    # to improve playback abilities
    frame_skip = 6

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break


        frame_idx += 1

        if frame_idx % frame_skip != 0:
            continue

        # Get current time in the video
        current_time_sec = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

        # --- 1. RUN THE AI ON THE FRAME ---
        # Force high-res scan (1280) and bump confidence slightly
        results = model.predict(source=frame, conf=0.15, imgsz=640, device="mps", verbose=False)

        ball_in_airspace = False

        # --- 2. CHECK WHAT IT FOUND ---
        if len(results[0].boxes) > 0:
            for box in results[0].boxes:
                
                # Check exactly what the AI thinks this is
                class_id = int(box.cls[0].item())
                class_name = model.names[class_id]
                confidence = box.conf[0].item()
                
                # If the AI found a person, shoe, or net... ignore it!
                if "ball" not in class_name.lower():
                    continue 
                
                # Print exactly what it found to your terminal
            #    print(f"Found: {class_name} ({confidence:.2f}) at {current_time_sec:.1f}s")
                
                # Get the ball's coordinates
                x1, y1, x2, y2 = box.xyxy[0].int().tolist()
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                
                # Draw a red dot on the ball
            #    cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

                # Math check: Is the red dot inside the blue airspace box?
                if (x <= cx <= x + w) and (y <= cy <= y + h):
                    ball_in_airspace = True
                    # Draw a green box around the ball
            #        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    break 

        # --- 3. DO SOMETHING IF THE BALL IS IN THE AIRSPACE ---
        if ball_in_airspace:
            print(f"🏐 BALL IN AIRSPACE at {current_time_sec:.1f}s!")
            active_seconds.append(current_time_sec) # --- NEW: Actually record the timestamp! ---
            # Trigger your Arduino/LED here if you are using one

        # --- 4. DRAW THE SCREEN AND SHOW IT --- (TURN ON FOR TESTING)
        # Draw your Airspace ROI rectangle
    #    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
    #    cv2.putText(frame, "Airspace", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    #    cv2.imshow("Volleyball Airspace Tracker", frame)
        
        # Press 'q' to quit the video early
    #    if cv2.waitKey(1) & 0xFF == ord('q'):
    #        break

        
        

    cap.release()
    cv2.destroyAllWindows()

    if not active_seconds:
        print("No ball detected. The model might need a lower confidence threshold.")
        return []

    # Group timestamps into rallies
    rallies = []
    rally_start = active_seconds[0]
    prev_sec = active_seconds[0]

    for sec in active_seconds[1:]:
        if sec - prev_sec > max_gap_sec:
            rally_end = prev_sec
            if (rally_end - rally_start) >= min_rally_sec:
                rallies.append((max(0, rally_start - buffer_sec), rally_end + buffer_sec))
            rally_start = sec
        prev_sec = sec

    if (prev_sec - rally_start) >= min_rally_sec:
        rallies.append((max(0, rally_start - buffer_sec), prev_sec + buffer_sec))

    return rallies

def export_clips(input_video, rallies, output_folder="V3.2court_rallies"):
    base_name = os.path.splitext(os.path.basename(input_video))[0]
    output_folder = os.path.join(output_folder, base_name)

    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)

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
    video_path = "/Users/seankang/Documents/vball/0802_games/0802_game_5.mp4"
    
    detected_rallies = analyze_airspace_with_custom_yolo(
        video_path, 
        start_sec=0,
        model_path="volleyball_yolo.pt", 
        min_rally_sec=3,   
        max_gap_sec=3.5,     
        buffer_sec=2.5       
    )

    if detected_rallies:
        print(f"\nFound {len(detected_rallies)} rallies!")
        export_clips(video_path, detected_rallies)