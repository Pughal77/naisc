import os
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from matplotlib.animation import FuncAnimation
import cv2


def load_tracking_summary(summary_file):
    """
    Load a tracking summary JSON file.

    Args:
        summary_file (str): Path to the tracking summary JSON file

    Returns:
        dict: Tracking summary data
    """
    with open(summary_file, 'r') as f:
        return json.load(f)


def visualize_object_durations(summary, output_dir):
    """
    Visualize the duration of each tracked object in the video.

    Args:
        summary (dict): Tracking summary data
        output_dir (str): Directory to save visualization
    """
    # Extract object durations
    durations = summary['object_durations']

    # Convert to DataFrame for easier plotting
    df = pd.DataFrame(durations)

    # Sort by duration
    df = df.sort_values('duration', ascending=False)

    # Create figure
    plt.figure(figsize=(12, 8))

    # Plot durations by class
    sns.barplot(x='object_id', y='duration', hue='class_name', data=df)

    plt.title(f"Object Durations in {summary['video_name']}")
    plt.xlabel("Object ID")
    plt.ylabel("Duration (seconds)")
    plt.xticks(rotation=90)
    plt.tight_layout()

    # Save figure
    output_file = os.path.join(
        output_dir, f"{summary['video_name']}_object_durations.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved object durations visualization to {output_file}")


def visualize_object_presence(summary, output_dir):
    """
    Visualize when each object is present in the video as a timeline.

    Args:
        summary (dict): Tracking summary data
        output_dir (str): Directory to save visualization
    """
    # Extract object information
    tracked_objects = summary['tracked_objects']
    video_duration = summary['duration']

    # Prepare data for visualization
    object_data = []
    for obj_id, obj in tracked_objects.items():
        object_data.append({
            'object_id': int(obj_id),
            'class_name': obj['class_name'],
            'first_timestamp': obj['first_timestamp'],
            'last_timestamp': obj['last_timestamp'],
            'duration': obj['duration']
        })

    # Sort by class name and first timestamp
    object_data.sort(key=lambda x: (x['class_name'], x['first_timestamp']))

    # Create figure
    plt.figure(figsize=(14, 10))

    # Plot presence timeline
    y_ticks = []
    y_labels = []

    # Define colors for different classes
    classes = set(obj['class_name'] for obj in object_data)
    class_colors = {}
    cmap = plt.cm.get_cmap('tab20', len(classes))
    for i, cls in enumerate(sorted(classes)):
        class_colors[cls] = cmap(i)

    for i, obj in enumerate(object_data):
        plt.barh(i, obj['duration'], left=obj['first_timestamp'],
                 color=class_colors[obj['class_name']], alpha=0.8)
        y_ticks.append(i)
        y_labels.append(f"{obj['class_name']} (ID:{obj['object_id']})")

    plt.yticks(y_ticks, y_labels)
    plt.xlim(0, video_duration)
    plt.title(f"Object Presence Timeline for {summary['video_name']}")
    plt.xlabel("Time (seconds)")
    plt.grid(axis='x')
    plt.tight_layout()

    # Save figure
    output_file = os.path.join(
        output_dir, f"{summary['video_name']}_object_timeline.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved object presence timeline to {output_file}")


def visualize_interactions(summary, output_dir):
    """
    Visualize interactions between objects.

    Args:
        summary (dict): Tracking summary data
        output_dir (str): Directory to save visualization
    """
    # Extract interactions
    interactions = summary['interactions']

    if not interactions:
        print(f"No interactions found for {summary['video_name']}")
        return

    # Convert string keys to tuples
    interaction_data = []
    for key, events in interactions.items():
        obj_ids = [int(id) for id in key.split(',')]
        obj1_id, obj2_id = obj_ids

        # Get object classes
        obj1_class = summary['tracked_objects'].get(
            str(obj1_id), {}).get('class_name', 'unknown')
        obj2_class = summary['tracked_objects'].get(
            str(obj2_id), {}).get('class_name', 'unknown')

        for event in events:
            interaction_data.append({
                'obj1_id': obj1_id,
                'obj2_id': obj2_id,
                'obj1_class': event['obj1_class'],
                'obj2_class': event['obj2_class'],
                'timestamp': event['timestamp'],
                'distance': event['distance']
            })

    if not interaction_data:
        print(f"No interaction data found for {summary['video_name']}")
        return

    # Convert to DataFrame
    df = pd.DataFrame(interaction_data)

    # Create figure for interaction timeline
    plt.figure(figsize=(14, 8))

    # Group by object pairs
    interaction_pairs = df.groupby(
        ['obj1_id', 'obj2_id', 'obj1_class', 'obj2_class'])

    y_ticks = []
    y_labels = []

    for i, ((obj1_id, obj2_id, obj1_class, obj2_class), group) in enumerate(interaction_pairs):
        # Plot interaction events
        plt.scatter(group['timestamp'], [i] * len(group), s=50, alpha=0.7)

        # Connect consecutive interactions
        if len(group) > 1:
            plt.plot(group['timestamp'], [i] * len(group), '-', alpha=0.5)

        y_ticks.append(i)
        y_labels.append(f"{obj1_class}({obj1_id}) & {obj2_class}({obj2_id})")

    plt.yticks(y_ticks, y_labels)
    plt.title(f"Object Interactions in {summary['video_name']}")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Interacting Objects")
    plt.grid(True)
    plt.tight_layout()

    # Save figure
    output_file = os.path.join(
        output_dir, f"{summary['video_name']}_interactions.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved interaction visualization to {output_file}")


def create_tracking_animation(summary, frames_dir, output_dir):
    """
    Create an animation of tracked objects using the original frames.

    Args:
        summary (dict): Tracking summary data
        frames_dir (str): Directory containing original video frames
        output_dir (str): Directory to save animation
    """
    video_name = summary['video_name']
    tracked_objects = summary['tracked_objects']

    # Find frames for this video
    frame_files = sorted([
        os.path.join(frames_dir, f) for f in os.listdir(frames_dir)
        if f.endswith('.jpg') or f.endswith('.png')
    ])

    if not frame_files:
        print(f"No frames found for {video_name} in {frames_dir}")
        return

    # Define colors for different object IDs
    num_objects = len(tracked_objects)
    color_map = {}
    cmap = plt.cm.get_cmap('hsv', num_objects)

    for i, obj_id in enumerate(tracked_objects.keys()):
        color = tuple(int(255 * x) for x in cmap(i)[:3])  # Convert to BGR
        color_map[int(obj_id)] = color

    # Create output video
    first_frame = cv2.imread(frame_files[0])
    height, width = first_frame.shape[:2]

    output_file = os.path.join(output_dir, f"{video_name}_tracking.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_file, fourcc, 5, (width, height))  # 5 FPS

    # Process each frame
    for i, frame_file in enumerate(frame_files):
        # Load frame
        frame = cv2.imread(frame_file)

        # Find objects in this frame
        frame_idx = i  # Assuming frames are ordered by index

        # Draw bounding boxes and IDs for objects in this frame
        for obj_id, obj in tracked_objects.items():
            obj_id = int(obj_id)

            if frame_idx in obj['frames']:
                # Get object position in this frame
                frame_index_in_obj = obj['frames'].index(frame_idx)
                bbox = obj['bboxes'][frame_index_in_obj]

                # Convert bbox to int
                x1, y1, x2, y2 = map(int, bbox)

                # Draw bounding box
                color = color_map[obj_id]
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                # Draw ID and class
                text = f"{obj['class_name']} ID:{obj_id}"
                cv2.putText(frame, text, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                # Draw trail (last 10 positions)
                trail_length = min(10, len(obj['centroids']))
                if trail_length > 1 and frame_index_in_obj > 0:
                    trail_end = frame_index_in_obj + 1
                    trail_start = max(0, trail_end - trail_length)

                    for j in range(trail_start, trail_end - 1):
                        pt1 = tuple(map(int, obj['centroids'][j]))
                        pt2 = tuple(map(int, obj['centroids'][j+1]))
                        cv2.line(frame, pt1, pt2, color, 2)

        # Add frame counter
        timestamp = frame_idx * summary.get('frame_interval', 0.5)
        cv2.putText(frame, f"Frame: {frame_idx}, Time: {timestamp:.1f}s",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Write frame to video
        out.write(frame)

    # Release video writer
    out.release()

    print(f"Saved tracking animation to {output_file}")


def process_tracking_summaries(summaries_dir, output_dir, frames_dir=None):
    """
    Process all tracking summary files in a directory and generate visualizations.

    Args:
        summaries_dir (str): Directory containing tracking summary JSON files
        output_dir (str): Directory to save visualizations
        frames_dir (str): Directory containing frame images (optional)
    """
    os.makedirs(output_dir, exist_ok=True)

    # Find all summary files - looking in video subfolders now
    summary_files = []

    # First, check for video subfolders
    for item in os.listdir(summaries_dir):
        video_dir = os.path.join(summaries_dir, item)
        if os.path.isdir(video_dir):
            # Look for tracking_summary.json in each video folder
            summary_file = os.path.join(video_dir, "tracking_summary.json")
            if os.path.exists(summary_file):
                summary_files.append(summary_file)

    # If no files found in subfolders, try legacy format
    if not summary_files:
        for root, _, files in os.walk(summaries_dir):
            for file in files:
                if file.endswith('_tracking_summary.json'):
                    summary_files.append(os.path.join(root, file))

    print(f"Found {len(summary_files)} tracking summary files to process")

    # Process each summary file
    for summary_file in summary_files:
        print(f"Processing {summary_file}")

        summary = load_tracking_summary(summary_file)
        video_name = summary['video_name']

        # Create a video-specific output directory
        video_output_dir = os.path.join(output_dir, video_name)
        os.makedirs(video_output_dir, exist_ok=True)

        # Create visualizations
        visualize_object_durations(summary, video_output_dir)
        visualize_object_presence(summary, video_output_dir)
        visualize_interactions(summary, video_output_dir)

        # If frames directory is provided, create tracking animation
        if frames_dir:
            video_frames_dir = os.path.join(frames_dir, video_name)
            if os.path.exists(video_frames_dir):
                create_tracking_animation(
                    summary, video_frames_dir, video_output_dir)
            else:
                print(
                    f"Frames directory not found for {video_name}: {video_frames_dir}")

    print(f"Completed visualization for {len(summary_files)} videos")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Visualize object tracking results')
    parser.add_argument('--summaries_dir', type=str, required=True,
                        help='Directory containing tracking summary JSON files')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save visualizations')
    parser.add_argument('--frames_dir', type=str,
                        help='Directory containing frame images (optional)')

    args = parser.parse_args()

    process_tracking_summaries(
        args.summaries_dir, args.output_dir, args.frames_dir)
