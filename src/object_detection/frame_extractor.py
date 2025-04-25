import cv2
import os
import numpy as np
from pathlib import Path
import glob


def extract_frames(video_path, output_dir, interval=0.5, return_frames=False):
    """
    Extract frames from a video at specified time intervals.

    Args:
        video_path (str): Path to the video file
        output_dir (str): Directory to save extracted frames
        interval (float): Time interval between frames in seconds (default: 0.5)
        return_frames (bool): If True, return the extracted frames as a list

    Returns:
        list: List of extracted frames as numpy arrays if return_frames is True
        int: Number of frames extracted
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Error opening video file: {video_path}")

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = int(fps * interval)

    # Check if frame interval is valid
    if frame_interval <= 0:
        frame_interval = 1

    frames = []
    frame_count = 0
    saved_count = 0

    video_name = os.path.splitext(os.path.basename(video_path))[0]

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Extract frame at the specified interval
        if frame_count % frame_interval == 0:
            # Save frame
            frame_filename = os.path.join(
                output_dir, f"{video_name}_frame_{saved_count:05d}.jpg")
            cv2.imwrite(frame_filename, frame)

            if return_frames:
                frames.append(frame)

            saved_count += 1

        frame_count += 1

    cap.release()

    print(
        f"Extracted {saved_count} frames from {video_path} at {interval}s intervals")

    if return_frames:
        return frames, saved_count
    return saved_count


def extract_frames_from_directory(video_dir, output_dir, interval=0.5):
    """
    Extract frames from all videos in a directory.

    Args:
        video_dir (str): Directory containing video files
        output_dir (str): Directory to save extracted frames
        interval (float): Time interval between frames in seconds

    Returns:
        int: Total number of frames extracted
    """
    video_extensions = ["*.mp4", "*.avi", "*.mov", "*.mkv"]
    video_paths = []

    for ext in video_extensions:
        video_paths.extend(glob.glob(os.path.join(
            video_dir, "**", ext), recursive=True))

    print(f"Found {len(video_paths)} videos in {video_dir}")

    total_frames = 0
    for video_path in video_paths:
        # Create subdirectory for each video
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        video_output_dir = os.path.join(output_dir, video_name)

        frames_count = extract_frames(video_path, video_output_dir, interval)
        total_frames += frames_count

    return total_frames


if __name__ == "__main__":
    # Example usage
    dataset_path = "dataset/Benchmark-AllVideos-HQ-Encoded-challenge"
    output_path = "dataset/extracted_frames"

    total_frames = extract_frames_from_directory(
        dataset_path, output_path, interval=0.5)
    print(f"Total frames extracted: {total_frames}")
