# Temporal Object Tracking

This directory contains tools for temporal tracking of objects detected across video frames.

## Overview

The tracking system processes JSON detection files produced by the YOLOv8/YOLOv10 detector and tracks objects across frames using IoU (Intersection over Union) matching. It generates temporal metadata about each object, including:

- First appearance timestamp
- Last appearance timestamp
- Total duration in the video
- Position changes over time
- Interactions with other objects

## Files

- `track_objects.py`: Main object tracking script
- `visualize_tracking.py`: Visualization tools for tracking results

## Usage

### Step 1: Run Object Tracking

Process detection JSON files to track objects across frames:

```bash
python src/temporal_tracking/track_objects.py \
  --detection_dir dataset/yolov10_results \
  --output_dir dataset/tracking_results \
  --frame_interval 0.5
```

Arguments:

- `--detection_dir`: Directory containing detection JSON files (must end with `_detections.json`)
- `--output_dir`: Directory to save tracking results
- `--frame_interval`: Time interval between frames in seconds (default: 0.5)

This will generate tracking summary JSON files for each video with detailed temporal metadata.

### Step 2: Visualize Tracking Results

Generate visualizations from tracking results:

```bash
python src/temporal_tracking/visualize_tracking.py \
  --summaries_dir dataset/tracking_results \
  --output_dir dataset/tracking_visualizations \
  --frames_dir dataset/extracted_frames
```

Arguments:

- `--summaries_dir`: Directory containing tracking summary JSON files
- `--output_dir`: Directory to save visualizations
- `--frames_dir`: (Optional) Directory containing extracted video frames to create tracking animations

## Output

The tracking system produces the following outputs:

### Tracking Summary JSON

For each video, a tracking summary JSON file is created with:

- Video metadata (name, duration, frame count)
- List of tracked objects with IDs
- Temporal information for each object (timestamps, positions, etc.)
- Detected interactions between objects

### Visualizations

The visualization script produces:

1. **Object duration plots**: Bar charts showing how long each object appears
2. **Object presence timelines**: Timeline showing when each object appears/disappears
3. **Interaction plots**: Visualizations of when objects interact
4. **Tracking animations**: MP4 videos showing tracked objects with bounding boxes and trails (if frames are provided)

## Example

To process a specific video's detections:

```python
from track_objects import ObjectTracker

# Initialize tracker for a specific detection file
tracker = ObjectTracker('dataset/yolov10_results/video1/json_detections/video1_detections.json')

# Track objects
tracker.track_objects()

# Generate and save summary
summary = tracker.generate_summary('output_dir')

# Access tracked objects
print(f"Found {len(tracker.tracked_objects)} objects")
for obj_id, obj in tracker.tracked_objects.items():
    print(f"Object {obj_id} ({obj['class_name']}): Duration = {obj['duration']}s")
```

## Notes

- The tracking uses a frame-to-frame association approach based on IoU
- Objects of the same class that have high bounding box overlap between frames are considered the same object
- Interactions are detected based on proximity between object centroids
- The system handles brief occlusions by maintaining object identity for up to 5 frames
