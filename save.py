import os
import shutil
import numpy as np
from moviepy import VideoFileClip

def analyze_audio_excitement(clip_path):
    """
    Analyzes the audio and returns the average volume of the LOUDEST 1.5 seconds.
    Perfect for catching quiet rallies that end with a loud spike or cheer.
    """
    try:
        clip = VideoFileClip(clip_path)
        
        if clip.audio is None:
            clip.close()
            return 0.0
            
        audio_array = clip.audio.to_soundarray()
        sample_rate = clip.audio.fps
        
        abs_audio = np.abs(audio_array.flatten())
        sorted_audio = np.sort(abs_audio)
        
        top_samples_count = int(sample_rate * 1.5) 
        
        if len(sorted_audio) < top_samples_count:
            top_samples_count = len(sorted_audio)
            
        bang_score = np.mean(sorted_audio[-top_samples_count:]) 
        
        clip.close()
        return bang_score
    
    except Exception as e:
        print(f"Error analyzing {clip_path}: {e}")
        return 0.0

def filter_exciting_rallies(input_folder, output_folder):
    """
    Scans all video clips, calculates the average excitement score, 
    and copies only the clips that score ABOVE the average while appending 
    their score to the new file name.
    """
    if not os.path.exists(input_folder):
        print(f"Error: Input folder '{input_folder}' not found.")
        return

    os.makedirs(output_folder, exist_ok=True)
    
    clips = [f for f in os.listdir(input_folder) if f.endswith(".mp4")]
    
    if not clips:
        print("No .mp4 files found in the directory.")
        return

    print(f"Step 1: Scanning {len(clips)} clips to calculate the average excitement...\n")
    
    clip_data = []
    
    # First Pass: Score everything
    for filename in sorted(clips):
        clip_path = os.path.join(input_folder, filename)
        score = analyze_audio_excitement(clip_path)
        clip_data.append((filename, score, clip_path))
        print(f"Scored {filename}: {score:.4f}")

    # Calculate average
    all_scores = [data[1] for data in clip_data]
    average_score = np.mean(all_scores)
    
    print(f"\n--- Dynamic Average Score: {average_score:.4f} ---\n")
    print("Step 2: Copying clips that beat the average...\n")
    
    kept_clips_count = 0

    # Second Pass: Filter, rename, and copy
    for filename, score, clip_path in clip_data:
        if score > average_score:
            # Separate the file name and extension (e.g., 'rally_1' and '.mp4')
            base_name, ext = os.path.splitext(filename)
            
            # Construct the new filename with the excitement score included
            new_filename = f"{base_name}_score_{score:.4f}{ext}"
            
            print(f"  -> HIGHLIGHT DETECTED! Saving as: {new_filename} 🔥")
            
            dest_path = os.path.join(output_folder, new_filename)
            shutil.copy2(clip_path, dest_path)
            kept_clips_count += 1
        else:
            print(f"     Skipping {filename} ({score:.4f})")

    print(f"\nDone! Kept {kept_clips_count} out of {len(clips)} clips.")
    print(f"Exciting clips saved to: {output_folder}")

if __name__ == "__main__":
    INPUT_DIR = "/Users/seankang/Documents/vball/V3.2court_rallies/0802_game_7" 
    BASE_OUTPUT_DIR = "/Users/seankang/Documents/vball/clips/0802_clips"
    
    game_analyzed = os.path.basename(INPUT_DIR)
    OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, game_analyzed)
    
    filter_exciting_rallies(INPUT_DIR, OUTPUT_DIR)