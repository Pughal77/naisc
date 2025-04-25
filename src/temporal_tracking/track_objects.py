import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from collections import defaultdict
import argparse
import cv2


class ObjectTracker:
    """
    Tracks objects across video frames based on detection data.
    Uses IoU (Intersection over Union) for object association.
    """

    def __init__(self, detections_file, frame_interval=0.5, iou_threshold=0.3):
        """
        Initialize the object tracker.

        Args:
            detections_file (str): Path to the JSON file with object detections
            frame_interval (float): Time interval between frames in seconds
            iou_threshold (float): Threshold for IoU matching
        """
        self.detections_file = detections_file
        self.frame_interval = frame_interval
        self.iou_threshold = iou_threshold

        # Video name from the detection file
        self.video_name = Path(detections_file).stem.replace('_detections', '')

        # Load detections
        with open(detections_file, 'r') as f:
            self.detections = json.load(f)

        # Convert string keys to integers
        self.detections = {int(k): v for k, v in self.detections.items()}

        # Sort frames
        self.frame_indices = sorted(list(map(int, self.detections.keys())))

        # Object state dictionary: {object_id: {info}}
        self.tracked_objects = {}

        # Next available object ID
        self.next_object_id = 0

    def calculate_iou(self, bbox1, bbox2):
        """
        Calculate IoU between two bounding boxes.

        Args:
            bbox1: [x1, y1, x2, y2]
            bbox2: [x1, y1, x2, y2]

        Returns:
            float: IoU value
        """
        # Calculate intersection area
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])

        if x2 < x1 or y2 < y1:
            return 0.0

        intersection_area = (x2 - x1) * (y2 - y1)

        # Calculate areas of both bounding boxes
        bbox1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        bbox2_area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])

        # Calculate union area
        union_area = bbox1_area + bbox2_area - intersection_area

        return intersection_area / union_area if union_area > 0 else 0.0

    def track_objects(self):
        """
        Perform object tracking across all frames.
        """
        # Process each frame in order
        for i, frame_idx in enumerate(self.frame_indices):
            frame_detections = self.detections[frame_idx]
            current_timestamp = frame_idx * self.frame_interval

            # Skip empty frames
            if not frame_detections:
                continue

            # First frame - assign new IDs to all objects
            if i == 0:
                for detection in frame_detections:
                    self._add_new_object(
                        detection, frame_idx, current_timestamp)
                continue

            # Create a list of current frame's bounding boxes
            current_bboxes = [detection['bbox']
                              for detection in frame_detections]
            current_classes = [detection['class_name']
                               for detection in frame_detections]

            # Create a list of previous frame's active objects
            # Only consider objects seen in the last 5 frames (to handle occlusions)
            active_objects = []
            for obj_id, obj_info in self.tracked_objects.items():
                if obj_info['last_frame'] >= self.frame_indices[max(0, i-5)]:
                    active_objects.append(obj_id)

            # Initialize assignment tracking
            assigned_detections = set()
            assigned_objects = set()

            # Calculate IoU between all active objects and new detections
            for obj_id in active_objects:
                obj_info = self.tracked_objects[obj_id]
                obj_bbox = obj_info['last_bbox']
                obj_class = obj_info['class_name']

                best_iou = self.iou_threshold
                best_detection_idx = None

                for j, (bbox, class_name) in enumerate(zip(current_bboxes, current_classes)):
                    if j in assigned_detections:
                        continue

                    # Only match objects of the same class
                    if class_name != obj_class:
                        continue

                    iou = self.calculate_iou(obj_bbox, bbox)
                    if iou > best_iou:
                        best_iou = iou
                        best_detection_idx = j

                # If we found a match, update the object
                if best_detection_idx is not None:
                    self._update_object(obj_id, frame_detections[best_detection_idx],
                                        frame_idx, current_timestamp)
                    assigned_detections.add(best_detection_idx)
                    assigned_objects.add(obj_id)

            # Create new objects for unassigned detections
            for j, detection in enumerate(frame_detections):
                if j not in assigned_detections:
                    self._add_new_object(
                        detection, frame_idx, current_timestamp)

        # Calculate final statistics
        self._calculate_object_stats()

        return self.tracked_objects

    def _add_new_object(self, detection, frame_idx, timestamp):
        """
        Add a new object to tracking.
        """
        obj_id = self.next_object_id
        self.next_object_id += 1

        self.tracked_objects[obj_id] = {
            'object_id': obj_id,
            'class_name': detection['class_name'],
            'class_id': detection['class_id'],
            'first_frame': frame_idx,
            'last_frame': frame_idx,
            'first_timestamp': timestamp,
            'last_timestamp': timestamp,
            'frames': [frame_idx],
            'timestamps': [timestamp],
            'bboxes': [detection['bbox']],
            'confidences': [detection['confidence']],
            'centroids': [self._calculate_centroid(detection['bbox'])],
            'last_bbox': detection['bbox']
        }

    def _update_object(self, obj_id, detection, frame_idx, timestamp):
        """
        Update an existing tracked object with new detection.
        """
        obj = self.tracked_objects[obj_id]

        # Update object information
        obj['last_frame'] = frame_idx
        obj['last_timestamp'] = timestamp
        obj['frames'].append(frame_idx)
        obj['timestamps'].append(timestamp)
        obj['bboxes'].append(detection['bbox'])
        obj['confidences'].append(detection['confidence'])
        obj['centroids'].append(self._calculate_centroid(detection['bbox']))
        obj['last_bbox'] = detection['bbox']

    def _calculate_centroid(self, bbox):
        """
        Calculate the centroid of a bounding box.
        """
        x1, y1, x2, y2 = bbox
        return [(x1 + x2) / 2, (y1 + y2) / 2]

    def _calculate_object_stats(self):
        """
        Calculate additional statistics for each tracked object.
        """
        for obj_id, obj in self.tracked_objects.items():
            # Calculate duration
            obj['duration'] = obj['last_timestamp'] - obj['first_timestamp']

            # Calculate bounding box area over time
            obj['areas'] = [(bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                            for bbox in obj['bboxes']]

            # Calculate distance traveled (using centroids)
            total_distance = 0
            for i in range(1, len(obj['centroids'])):
                prev_centroid = obj['centroids'][i-1]
                curr_centroid = obj['centroids'][i]
                # Euclidean distance
                distance = np.sqrt((curr_centroid[0] - prev_centroid[0])**2 +
                                   (curr_centroid[1] - prev_centroid[1])**2)
                total_distance += distance

            obj['total_distance'] = total_distance

            # Calculate average speed (pixels per second)
            if obj['duration'] > 0:
                obj['avg_speed'] = total_distance / obj['duration']
            else:
                obj['avg_speed'] = 0

    def detect_interactions(self, distance_threshold=100):
        """
        Detect potential interactions between objects based on proximity.

        Args:
            distance_threshold (float): Maximum distance to consider an interaction

        Returns:
            dict: Dictionary of interactions {(obj_id1, obj_id2): [frame_indices]}
        """
        interactions = defaultdict(list)

        # For each frame, check for objects that are close to each other
        for frame_idx in self.frame_indices:
            # Get all objects in current frame
            objects_in_frame = []

            for obj_id, obj in self.tracked_objects.items():
                if frame_idx in obj['frames']:
                    frame_index_in_obj = obj['frames'].index(frame_idx)
                    centroid = obj['centroids'][frame_index_in_obj]
                    objects_in_frame.append({
                        'object_id': obj_id,
                        'class_name': obj['class_name'],
                        'centroid': centroid
                    })

            # Check for proximity between each pair of objects
            for i in range(len(objects_in_frame)):
                for j in range(i+1, len(objects_in_frame)):
                    obj1 = objects_in_frame[i]
                    obj2 = objects_in_frame[j]

                    # Calculate distance between centroids
                    distance = np.sqrt((obj1['centroid'][0] - obj2['centroid'][0])**2 +
                                       (obj1['centroid'][1] - obj2['centroid'][1])**2)

                    # Record interaction if distance is below threshold
                    if distance < distance_threshold:
                        interaction_key = tuple(
                            sorted([obj1['object_id'], obj2['object_id']]))
                        interactions[interaction_key].append({
                            'frame': frame_idx,
                            'timestamp': frame_idx * self.frame_interval,
                            'distance': distance,
                            'obj1_class': obj1['class_name'],
                            'obj2_class': obj2['class_name']
                        })

        return interactions

    def generate_summary(self, output_dir=None):
        """
        Generate a summary of the tracked objects.

        Args:
            output_dir (str): Directory to save summary files

        Returns:
            dict: Summary information
        """
        if output_dir:
            # Create main output directory
            os.makedirs(output_dir, exist_ok=True)

            # Create a video-specific subfolder
            video_output_dir = os.path.join(output_dir, self.video_name)
            os.makedirs(video_output_dir, exist_ok=True)

        # Aggregate tracked objects into a summary
        summary = {
            'video_name': self.video_name,
            'total_frames': len(self.frame_indices),
            'duration': self.frame_indices[-1] * self.frame_interval,
            'objects_by_class': defaultdict(int),
            'object_durations': [],
            'tracked_objects': self.tracked_objects
        }

        # Count objects by class and gather duration information
        for obj_id, obj in self.tracked_objects.items():
            summary['objects_by_class'][obj['class_name']] += 1
            summary['object_durations'].append({
                'object_id': obj_id,
                'class_name': obj['class_name'],
                'duration': obj['duration'],
                'first_timestamp': obj['first_timestamp'],
                'last_timestamp': obj['last_timestamp'],
                'frames_count': len(obj['frames'])
            })

        # Detect interactions
        interactions = self.detect_interactions()
        summary['interactions'] = interactions

        # Save summary to file if output_dir is provided
        if output_dir:
            summary_file = os.path.join(
                video_output_dir, "tracking_summary.json")

            # Remove defaultdict before saving
            summary_to_save = summary.copy()
            summary_to_save['objects_by_class'] = dict(
                summary_to_save['objects_by_class'])

            # Convert interactions keys to strings
            interactions_dict = {}
            for key, value in interactions.items():
                interactions_dict[f"{key[0]},{key[1]}"] = value
            summary_to_save['interactions'] = interactions_dict

            with open(summary_file, 'w') as f:
                json.dump(summary_to_save, f, indent=2)

            print(f"Saved tracking summary to {summary_file}")

            # Generate trajectory visualization
            self._visualize_trajectories(video_output_dir)

        return summary

    def _visualize_trajectories(self, output_dir):
        """
        Generate visualizations of object trajectories.
        """
        plt.figure(figsize=(12, 8))

        # Plot trajectories for each object
        for obj_id, obj in self.tracked_objects.items():
            if len(obj['centroids']) < 2:  # Skip objects that appear in only one frame
                continue

            # Extract x and y coordinates from centroids
            x_coords = [centroid[0] for centroid in obj['centroids']]
            y_coords = [centroid[1] for centroid in obj['centroids']]

            # Plot trajectory
            plt.plot(x_coords, y_coords, '-o',
                     label=f"{obj['class_name']} ID:{obj_id}")

            # Mark first and last positions
            plt.plot(x_coords[0], y_coords[0], 'go',
                     markersize=10)  # Green for start
            plt.plot(x_coords[-1], y_coords[-1], 'ro',
                     markersize=10)  # Red for end

        plt.gca().invert_yaxis()  # Invert y-axis to match image coordinates
        plt.title(f"Object Trajectories for {self.video_name}")
        plt.xlabel("X Coordinate")
        plt.ylabel("Y Coordinate")
        plt.legend(loc='best')
        plt.grid(True)

        # Save the visualization
        trajectory_file = os.path.join(output_dir, "trajectories.png")
        plt.savefig(trajectory_file, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Saved trajectory visualization to {trajectory_file}")


def process_all_detection_files(detection_dir, output_dir, frame_interval=0.5):
    """
    Process all detection JSON files in a directory.

    Args:
        detection_dir (str): Directory containing detection JSON files
        output_dir (str): Directory to save tracking results
        frame_interval (float): Time interval between frames in seconds
    """
    os.makedirs(output_dir, exist_ok=True)

    # Find all detection JSON files
    detection_files = []
    for root, _, files in os.walk(detection_dir):
        for file in files:
            if file.endswith('_detections.json'):
                detection_files.append(os.path.join(root, file))

    print(f"Found {len(detection_files)} detection files to process")

    all_results = {}

    # Process each detection file
    for detection_file in detection_files:
        print(f"Processing {detection_file}")

        tracker = ObjectTracker(detection_file, frame_interval=frame_interval)
        tracker.track_objects()
        summary = tracker.generate_summary(output_dir)

        video_name = Path(detection_file).stem.replace('_detections', '')
        all_results[video_name] = summary

    # Save aggregate summary
    aggregate_summary = {
        'processed_videos': len(all_results),
        'videos': list(all_results.keys())
    }

    with open(os.path.join(output_dir, 'aggregate_tracking_summary.json'), 'w') as f:
        json.dump(aggregate_summary, f, indent=2)

    print(f"Completed tracking for {len(all_results)} videos")
    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Track objects across video frames')
    parser.add_argument('--detection_dir', type=str, required=True,
                        help='Directory containing detection JSON files')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save tracking results')
    parser.add_argument('--frame_interval', type=float, default=0.5,
                        help='Time interval between frames in seconds (default: 0.5)')

    args = parser.parse_args()

    process_all_detection_files(
        args.detection_dir, args.output_dir, args.frame_interval)
