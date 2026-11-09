import os
import shutil
import numpy as np
from moviepy import VideoFileClip

def analyze_audio_excitement(clip_path, sample_window_sec=1.5):
    """
    Analyzes the audio and returns the average volume of the loudest 1.5 seconds.
    """
    try:
        clip = VideoFileClip(clip_path)
        
        if clip.audio is None:
            clip.close()
            return 0.0
            
        audio_array = clip.audio.to_soundarray()
        sample_rate = clip.audio.fps
        
        # Merge stereo channels and take absolute amplitude
        abs_audio = np.abs(audio_array.flatten())
        
        # Sort samples quietest to loudest
        sorted_audio = np.sort(abs_audio)
        
        # Target the top X seconds of sound
        top_samples_count = int(sample_rate * sample_window_sec) 
        
        if len(sorted_audio) < top_samples_count:
            top_samples_count = len(sorted_audio)
            
        bang_score = np.mean(sorted_audio[-top_samples_count:])
        
        clip.close()
        return float(bang_score)
    
    except Exception as e:
        print(f"Error processing {os.path.basename(clip_path)}: {e}")
        return 0.0

def get_top_highlights(root_dir, output_dir, top_n=5):
    """
    Recursively scans all subfolders in root_dir, ranks every MP4 by audio excitement,
    and copies the top N clips to output_dir.
    """
    if not os.path.exists(root_dir):
        print(f"Error: Root folder '{root_dir}' does not exist.")
        return

    all_analyzed_clips = []
    
    print(f"Searching through all subfolders in:\n  {root_dir}\n")

    # os.walk recursively inspects every folder and subfolder
    for current_path, subdirs, files in os.walk(root_dir):
        # Ignore the output folder if it happens to sit inside root_dir
        if os.path.abspath(current_path).startswith(os.path.abspath(output_dir)):
            continue

        mp4_files = [f for f in files if f.endswith(".mp4")]
        
        for filename in mp4_files:
            clip_path = os.path.join(current_path, filename)
            parent_folder = os.path.basename(current_path)
            
            print(f"Analyzing [{parent_folder}] -> {filename}...")
            score = analyze_audio_excitement(clip_path)
            
            # Save score alongside the file path and its game folder name
            all_analyzed_clips.append({
                'score': score,
                'path': clip_path,
                'filename': filename,
                'folder': parent_folder
            })

    if not all_analyzed_clips:
        print("No .mp4 files found.")
        return

    # Sort all clips across all folders from highest score to lowest
    all_analyzed_clips.sort(key=lambda x: x['score'], reverse=True)
    
    top_clips = all_analyzed_clips[:top_n]

    # Display Leaderboard
    print("\n" + "="*50)
    print(f" 🔥 TOP {len(top_clips)} LOUDEST RALLIES OVERALL 🔥")
    print("="*50)
    for rank, item in enumerate(top_clips, start=1):
        print(f"#{rank} | Score: {item['score']:.4f} | Source: [{item['folder']}] {item['filename']}")
    print("="*50 + "\n")

    # Copy top clips to destination folder
    os.makedirs(output_dir, exist_ok=True)
    
    for rank, item in enumerate(top_clips, start=1):
        # Add rank prefix so files stay organized in order (e.g., "Rank_01_0726_game_9_rally_02.mp4")
        new_filename = f"Rank_{rank:02d}_{item['folder']}_{item['filename']}"
        dest_path = os.path.join(output_dir, new_filename)
        
        shutil.copy2(item['path'], dest_path)
        print(f"Copied Rank #{rank} to -> {dest_path}")

    print(f"\nDone! Top {len(top_clips)} highlights exported to: {output_dir}")

if __name__ == "__main__":
    # Point this to your main clips directory that holds all game folders
    ROOT_CLIPS_DIR = "/Users/seankang/Documents/vball/clips/0726_clips"
    
    game_analyzed = os.path.basename(ROOT_CLIPS_DIR)

    # Destination folder for the top overall highlights
    BASE_OUTPUT_TOP_DIR = "/Users/seankang/Documents/vball/clips/top_5_highlights"
    
    OUTPUT_TOP_DIR = os.path.join(BASE_OUTPUT_TOP_DIR, game_analyzed)
    # How many of the absolute loudest clips to keep
    TOP_N_COUNT = 5
    
    get_top_highlights(ROOT_CLIPS_DIR, OUTPUT_TOP_DIR, top_n=TOP_N_COUNT)