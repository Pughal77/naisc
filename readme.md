# AI Scene Comprehension Pipeline

A comprehensive pipeline for video scene analysis using object detection, temporal tracking, and large language models to generate detailed scene comprehension.

## Overview

This project processes short videos through several stages:

1. **Object Detection** - Detecting objects in video frames using YOLOv10
2. **Temporal Tracking** - Tracking objects across frames to understand movement and interactions
3. **Prompt Generation** - Creating structured prompts for multimodal LLMs
4. **LLM Querying** - Querying large language models to answer questions about the video scenes

## Dataset

The pipeline starts with the AISG Challenge dataset (not included in the repository):

- Downloaded from Hugging Face: [lmms-lab/AISG_Challenge](https://huggingface.co/datasets/lmms-lab/AISG_Challenge)
- Contains:
  - `Benchmark-AllVideos-HQ-Encoded-challenge/` - Video files for analysis
  - `metadata.parquet` - Questions and metadata for each video

## Project Structure

```
NAISC_project/
├── src/
│   ├── object_detection/      # YOLOv10 implementation
│   ├── temporal_tracking/     # Tracking objects across frames
│   ├── LLM_prompt_generation/ # Generating and querying prompts
│   └── readme.md              # Source code documentation
├── dataset/
│   ├── tracking_visualizations/ # Generated tracking visualizations
│   ├── tracking_results/        # JSON files with tracking data
│   ├── yolov10_results/         # Object detection results
│   ├── extracted_frames/        # Extracted video frames
│   ├── metadata.parquet         # From AISG Challenge dataset
│   └── Benchmark-AllVideos-HQ-Encoded-challenge/ # Original videos
└── exploratory_data_analysis/   # Analysis notebooks and scripts
```

## Pipeline Steps

### 1. Object Detection

Using a pre-trained YOLOv10 model, we detect objects in each frame of the videos:

- Processes videos from the Benchmark directory
- Extracts frames at regular intervals
- Detects objects, bounding boxes, and confidence scores
- Outputs detection results to `dataset/yolov10_results/`

### 2. Temporal Tracking

After object detection, we track objects across frames:

- Uses detection data from YOLOv10
- Implements tracking algorithms to maintain object identity
- Generates tracking visualizations (images showing tracked objects)
- Stores tracking data in `dataset/tracking_results/` as JSON files

### 3. Prompt Generation

We generate prompts for LLMs in two formats:

- **Standard prompts**: Include tracking data, visualizations, and questions
- **Lightweight prompts**: Minimal prompts with only tracking data

The prompt generation:

- Uses questions from `metadata.parquet`
- Incorporates tracking data and visualizations
- Structures data for multimodal or text-only LLMs

### 4. LLM Querying

Finally, we query LLMs to analyze the scenes:

- Options for both API-based querying (Ollama, LM Studio) and direct model loading
- The `query_llm.py` script loads models directly from Hugging Face
- Results are saved as structured JSON files

## Running the Pipeline

### Environment Setup

```bash
# Clone the repository
git clone https://github.com/your-username/NAISC_project.git
cd NAISC_project

# Install dependencies
pip install -r requirements.txt
```

### Object Detection

```bash
python src/object_detection/detect_objects.py \
  --videos_dir dataset/Benchmark-AllVideos-HQ-Encoded-challenge \
  --output_dir dataset/yolov10_results
```

### Temporal Tracking

```bash
python src/temporal_tracking/track_objects.py \
  --detection_dir dataset/yolov10_results \
  --output_dir dataset/tracking_results \
  --vis_dir dataset/tracking_visualizations
```

### Prompt Generation

```bash
# Standard prompts
python src/LLM_prompt_generation/generate_prompts.py \
  --vis_dir dataset/tracking_visualizations \
  --tracking_dir dataset/tracking_results \
  --yolo_dir dataset/yolov10_results \
  --metadata_file dataset/metadata.parquet \
  --output_dir output/prompts

# Lightweight prompts
python src/LLM_prompt_generation/generate_lightweight_prompts.py \
  --tracking_dir dataset/tracking_results \
  --metadata_file dataset/metadata.parquet \
  --output_dir output/lightweight_prompts
```

### LLM Querying

```bash
# Query using direct Hugging Face model loading
python src/LLM_prompt_generation/query_llm.py \
  --prompt_dir output/prompts \
  --output_dir output/responses \
  --model_name mistralai/Mistral-7B-Instruct-v0.2

# Query using external API (Ollama, LM Studio)
python src/LLM_prompt_generation/query_local_llm.py \
  --prompt_dir output/prompts \
  --output_dir output/responses \
  --model_type ollama \
  --model_name llava
```

## Acknowledgments

- [AISG Challenge dataset](https://huggingface.co/datasets/lmms-lab/AISG_Challenge) for providing the video data and question framework
- YOLOv10 for state-of-the-art object detection capabilities
- Hugging Face for model hosting and infrastructure
