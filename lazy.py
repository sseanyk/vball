import os
import shutil

def organize_files_by_filename(source_dir, output_dir, keywords):
    """
    Scans filenames in source_dir and its subfolders for specific keywords.
    Copies matching files into keyword-specific folders inside output_dir.
    """
    # Create the main output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Convert keywords to lowercase for case-insensitive matching
    search_keywords = [kw.lower() for kw in keywords]

    # Walk through the source directory and all nested folders
    for root, _, files in os.walk(source_dir):
        for file in files:
            file_path = os.path.join(root, file)
            
            # Convert the filename to lowercase for checking
            filename_lower = file.lower()

            # Check each keyword against the filename
            for original_kw, search_kw in zip(keywords, search_keywords):
                if search_kw in filename_lower:
                    # Create a folder for this specific keyword
                    keyword_folder = os.path.join(output_dir, original_kw)
                    os.makedirs(keyword_folder, exist_ok=True)

                    try:
                        # Copy the file to the new folder
                        shutil.copy2(file_path, keyword_folder)
                        print(f"Copied: '{file}' -> '{original_kw}' folder.")
                    except Exception as e:
                        print(f"Could not copy '{file}': {e}")

# ==========================================
# Configuration Area
# ==========================================

# 1. The folder you want to search through
SOURCE_FOLDER = "/Users/seankang/Documents/vball/clips/0802_clips" 

# 2. The folder where the new keyword folders will be created
OUTPUT_FOLDER = "/Users/seankang/Documents/vball/clips/postable"

# 3. Add as many keywords as you want to this list
KEYWORDS_TO_FIND = [
    "maybe",
    "clip",
    "lowlight"
]

# Run the program
if __name__ == "__main__":
    print("Starting search by filename...")
    organize_files_by_filename(SOURCE_FOLDER, OUTPUT_FOLDER, KEYWORDS_TO_FIND)
    print("Search and organization complete!")