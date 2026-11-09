import cv2
import os

video_path = "0728.mp4"
output_folder = "training_images"
os.makedirs(output_folder, exist_ok=True)

# --- NEW: SET YOUR TIMESTAMPS (IN SECONDS) ---
start_time_sec = 215  
end_time_sec = 365   
# ---------------------------------------------

cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)

# Calculate which frame numbers correspond to your timestamps
start_frame = int(start_time_sec * fps)
end_frame = int(end_time_sec * fps)

# FAST FORWARD the video directly to the start frame
cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

# Keep track of our position
frame_count = start_frame 
saved_count = 0

print(f"Extracting frames from {start_time_sec}s to {end_time_sec}s...")

while cap.isOpened():
    ret, frame = cap.read()
    
    # Stop if the video ends OR if we pass our end time
    if not ret or frame_count > end_frame:
        break
    
    # Save 1 frame every 0.5 seconds
    if frame_count % int(fps * 0.5) == 0:
        cv2.imwrite(f"{output_folder}/frame_{saved_count:04d}.jpg", frame)
        saved_count += 1
        
    frame_count += 1

cap.release()
print(f"Done! Saved {saved_count} images to '{output_folder}'.")