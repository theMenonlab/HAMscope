import cv2
import os

def extract_frames(input_dir, output_dir):
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Loop through each .avi file (0.avi to 5.avi)
    for movie_num in range(6):
        movie_path = os.path.join(input_dir, f"{movie_num}.avi")
        
        # Check if the movie file exists
        if not os.path.isfile(movie_path):
            print(f"Warning: {movie_path} doesn't exist, skipping.")
            continue
        
        # Open the video file
        cap = cv2.VideoCapture(movie_path)
        
        if not cap.isOpened():
            print(f"Error: Could not open {movie_path}")
            continue
        
        frame_count = 0
        
        # Read and save each frame
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert frame to grayscale
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Save the grayscale frame with movie number in filename
            output_path = os.path.join(output_dir, f"movie_{movie_num}_frame_{frame_count:06d}.png")
            cv2.imwrite(output_path, gray_frame)
            
            frame_count += 1
            
            if frame_count % 100 == 0:
                print(f"Processed {frame_count} frames from movie {movie_num}")
        
        cap.release()
        print(f"Completed movie {movie_num}: Extracted {frame_count} frames")

# Hard-coded directories
input_directory = "/media/al/Extreme SSD/20250410_dasmeet/mut/miniscope/2025_04_10/14_30_52/miniscopeDeviceName"
output_directory = "/media/al/Extreme SSD/20250410_dasmeet/extracted_frames"

# Execute the extraction
extract_frames(input_directory, output_directory)
print("All movies processed!")