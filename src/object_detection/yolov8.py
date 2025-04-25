import os
import torch
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from tqdm import tqdm
import time
import glob
import json

from frame_extractor import extract_frames, extract_frames_from_directory


class YOLOv10Detector:
    """
    Object detection model using YOLOv10 for video frame analysis.
    """

    def __init__(self, model_size='n', device=None):
        """
        Initialize YOLOv10 model.

        Args:
            model_size (str): Size of the model ('n', 's', 'm', 'l', 'x')
            device (str): Device to run the model on ('cuda', 'cpu')
        """
        self.model_size = model_size

        # Auto-detect device if not specified
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        print(f"Using device: {self.device}")

        # Load pretrained YOLOv10 model
        try:
            # For YOLOv10, check the correct weight filename pattern
            model_path = f"yolov10{model_size}"
            self.model = YOLO(model_path)
            print(f"Loaded YOLOv10{model_size} model")
        except Exception as e:
            # If YOLOv10 is not available yet through ultralytics, default to YOLOv8
            print(f"Error loading YOLOv10: {e}")
            print("Falling back to YOLOv8...")
            model_path = f"yolov8{model_size}"
            self.model = YOLO(model_path)
            print(f"Loaded YOLOv8{model_size} model")

    def detect_objects(self, image, conf_threshold=0.25, iou_threshold=0.45):
        """
        Detect objects in an image.

        Args:
            image: Image as numpy array (BGR)
            conf_threshold (float): Confidence threshold for detections
            iou_threshold (float): IoU threshold for NMS

        Returns:
            List of detections (each with bbox, class_id, confidence)
        """
        # Run inference
        results = self.model(image, conf=conf_threshold, iou=iou_threshold)
        return results[0]  # Return first result (only one image)

    def process_frame(self, frame, save_path=None, show=False):
        """
        Process a single frame for object detection.

        Args:
            frame: Input frame as numpy array
            save_path (str): Path to save the annotated frame
            show (bool): Whether to display the annotated frame

        Returns:
            dict: Detection results
        """
        # Detect objects
        results = self.detect_objects(frame)

        # Get annotated frame with bounding boxes
        annotated_frame = results.plot()

        # Save if requested
        if save_path:
            cv2.imwrite(save_path, annotated_frame)

        # Show if requested
        if show:
            cv2.imshow("YOLOv10 Detection", annotated_frame)
            cv2.waitKey(1)

        # Extract detection information (class, confidence, bbox)
        detections = []

        if results.boxes is not None:
            for box in results.boxes:
                cls_id = int(box.cls.item())
                cls_name = results.names[cls_id]
                conf = box.conf.item()
                xyxy = box.xyxy.tolist()[0]  # Convert to list

                detection = {
                    'class_id': cls_id,
                    'class_name': cls_name,
                    'confidence': conf,
                    'bbox': xyxy  # [x1, y1, x2, y2]
                }
                detections.append(detection)

        return {
            'frame': annotated_frame,
            'detections': detections
        }

    def process_video(self, video_path, output_dir, interval=0.5, save_frames=True):
        """
        Process a video file, extract frames and perform object detection.
        If frames have already been extracted, use those instead of re-extracting.

        Args:
            video_path (str): Path to the video file
            output_dir (str): Directory to save results
            interval (float): Interval in seconds between frames to process
            save_frames (bool): Whether to save the annotated frames

        Returns:
            dict: Detection results for all processed frames
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Create detection results directory
        detection_dir = os.path.join(output_dir, "detections")
        os.makedirs(detection_dir, exist_ok=True)

        # Extract frames from video
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        frames_dir = os.path.join(
            # ← to go to dataset directory
            os.path.dirname(os.path.dirname(video_path)),
            "extracted_frames",
            video_name
        )

        # Check if frames have already been extracted
        frames = []
        if os.path.exists(frames_dir) and len(os.listdir(frames_dir)) > 0:
            print(
                f"Found existing extracted frames for {video_name}. Using these instead of re-extracting.")
            # Load existing frames
            frame_files = sorted(glob.glob(os.path.join(
                frames_dir, f"{video_name}_frame_*.jpg")))
            if not frame_files:
                frame_files = sorted(
                    glob.glob(os.path.join(frames_dir, "*.jpg")))

            if frame_files:
                print(f"Found {len(frame_files)} existing frames.")
                frames = [cv2.imread(frame_file) for frame_file in frame_files]
                num_frames = len(frames)
            else:
                print(
                    f"No frame files found in {frames_dir}. Will extract frames.")
                frames, num_frames = extract_frames(
                    video_path, frames_dir, interval, return_frames=True)
        else:
            # Extract frames from video if they don't exist
            print(f"Extracting frames from {video_name}...")
            os.makedirs(frames_dir, exist_ok=True)
            frames, num_frames = extract_frames(
                video_path, frames_dir, interval, return_frames=True)

        # Process each frame
        all_results = {}
        for i, frame in enumerate(tqdm(frames, desc="Processing frames")):
            # Process the frame
            annotated_path = os.path.join(
                detection_dir, f"{video_name}_detection_{i:05d}.jpg") if save_frames else None
            result = self.process_frame(frame, save_path=annotated_path)

            # Store results
            all_results[i] = result['detections']

        # Save all detections for this video as one JSON file
        json_output_path = os.path.join(
            output_dir, "json_detections", f"{video_name}_detections.json")
        os.makedirs(os.path.dirname(json_output_path), exist_ok=True)
        with open(json_output_path, 'w') as f:
            json.dump(all_results, f, indent=2)

        return all_results

    def process_dataset(self, dataset_dir, output_dir, interval=0.5):
        """
        Process all videos in a dataset directory.

        Args:
            dataset_dir (str): Directory containing video files
            output_dir (str): Directory to save results
            interval (float): Interval between frames to process

        Returns:
            dict: Detection results for all videos
        """
        os.makedirs(output_dir, exist_ok=True)

        # Find all video files
        video_extensions = ["*.mp4", "*.avi", "*.mov", "*.mkv"]
        video_paths = []

        for ext in video_extensions:
            video_paths.extend(glob.glob(os.path.join(
                dataset_dir, "**", ext), recursive=True))

        print(f"Found {len(video_paths)} videos in {dataset_dir}")

        # Process each video
        all_results = {}
        for video_path in video_paths:
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            print(f"Processing video: {video_name}")

            video_output_dir = os.path.join(output_dir, video_name)
            results = self.process_video(
                video_path, video_output_dir, interval)

            all_results[video_name] = results

        return all_results


if __name__ == "__main__":
    # Example usage
    dataset_path = "dataset/Benchmark-AllVideos-HQ-Encoded-challenge"
    output_path = "dataset/yolov10_results"

    # Initialize detector
    detector = YOLOv10Detector(model_size='m')  # medium size model

    # Process the dataset
    results = detector.process_dataset(dataset_path, output_path, interval=0.5)

    print(f"Processing complete. Results saved to {output_path}")
