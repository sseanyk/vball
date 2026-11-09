import cv2
import os
import numpy as np
from collections import deque
from ultralytics import YOLO
from moviepy import VideoFileClip

def convert_to_vertical_adaptive_camera(
    input_video_path, 
    output_video_path="vertical_adaptive_pan.mp4", 
    model_path="volleyball_yolo.pt",
    base_smoothing=0.15,  # Speed for slow plays (cinematic float)
    max_smoothing=0.45,   # Speed for fast plays (aggressive whip pan)
    speed_threshold=75.0, # Pixels per frame required to trigger max speed
    conf_threshold=0.15,
    target_aspect_ratio=9/16,
    drift_timeout_frames=15 
):
    if not os.path.exists(input_video_path):
        print(f"Error: Input video '{input_video_path}' not found.")
        return

    print(f"Loading YOLO Model from '{model_path}'...")
    model = YOLO(model_path)

    cap = cv2.VideoCapture(input_video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    crop_h = orig_h
    crop_w = int(crop_h * target_aspect_ratio)
    if crop_w > orig_w:
        crop_w = orig_w

    temp_output_path = "temp_cropped_no_audio.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_output_path, fourcc, fps, (crop_w, crop_h))

    center_of_court = orig_w / 2.0
    current_crop_center_x = center_of_court
    target_x = center_of_court
    prev_target_x = center_of_court
    
    # The Shock Absorber
    recent_x_positions = deque(maxlen=5) 
    frames_since_last_seen = 0
    frame_count = 0

    print("\nProcessing frames with Adaptive Camera engine...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % 30 == 0:
            print(f"Processing frame {frame_count}/{total_frames}...")

        results = model.predict(source=frame, conf=conf_threshold, imgsz=640, device="mps", verbose=False)
        ball_found = False

        if len(results[0].boxes) > 0:
            for box in results[0].boxes:
                class_id = int(box.cls[0].item())
                if "ball" in model.names[class_id].lower():
                    x1, y1, x2, y2 = box.xyxy[0].int().tolist()
                    ball_x = (x1 + x2) // 2
                    
                    recent_x_positions.append(ball_x)
                    target_x = np.median(recent_x_positions)
                    
                    ball_found = True
                    frames_since_last_seen = 0
                    break

        if not ball_found:
            frames_since_last_seen += 1
            if frames_since_last_seen > drift_timeout_frames:
                target_x = center_of_court
                recent_x_positions.clear()

        # --- NEW: ADAPTIVE SMOOTHING ENGINE ---
        # 1. Calculate how fast the target is moving
        pixel_speed = abs(target_x - prev_target_x)
        
        # 2. Convert speed to a percentage (0.0 to 1.0)
        # If it moves 50 pixels and threshold is 100, intensity is 0.5 (50%)
        speed_intensity = min(pixel_speed / speed_threshold, 1.0)
        
        # 3. Scale the smoothing factor based on intensity
        dynamic_smoothing = base_smoothing + ((max_smoothing - base_smoothing) * speed_intensity)
        
        # Save current target for the next frame's speed calculation
        prev_target_x = target_x 

        # Apply camera movement using the calculated dynamic smoothing
        current_crop_center_x += (target_x - current_crop_center_x) * dynamic_smoothing

        crop_x1 = int(current_crop_center_x - (crop_w / 2))

        # Clamp to edges
        if crop_x1 < 0:
            crop_x1 = 0
        elif crop_x1 + crop_w > orig_w:
            crop_x1 = orig_w - crop_w

        crop_x2 = crop_x1 + crop_w
        cropped_frame = frame[0:crop_h, crop_x1:crop_x2]
        out.write(cropped_frame)

    cap.release()
    out.release()

    print("\nAttaching original audio track...")
    try:
        video_clip = VideoFileClip(temp_output_path)
        original_clip = VideoFileClip(input_video_path)

        if original_clip.audio is not None:
            final_clip = video_clip.with_audio(original_clip.audio)
            final_clip.write_videofile(output_video_path, codec="libx264", audio_codec="aac", logger=None)
            final_clip.close()
        else:
            os.rename(temp_output_path, output_video_path)

        video_clip.close()
        original_clip.close()

        if os.path.exists(temp_output_path):
            os.remove(temp_output_path)

        print(f"\n🎉 Finished! Saved adaptive vertical video to: {output_video_path}")

    except Exception as e:
        print(f"Error transferring audio: {e}")

if __name__ == "__main__":
    convert_to_vertical_adaptive_camera(
        input_video_path="/Users/seankang/Documents/vball/clips/0726_clips/postable/hits/0726_game_8_rally_12_(401.9s-414.5s) (soham).mp4",
        output_video_path="/Users/seankang/Documents/vball/clips/0726_clips/vertical_predictive.mp4",
        model_path="volleyball_yolo.pt"
    )

