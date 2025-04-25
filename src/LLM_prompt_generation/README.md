# LLM Prompt Generation and Analysis

This directory contains scripts for generating prompts from tracking, visualization, and detection data, querying locally-hosted multimodal LLMs, and saving the responses for traffic scene analysis.

## Overview

The pipeline consists of several components:

1. **Data Collection**: Tracking results, tracking visualizations, and YOLOv10 detection results
2. **Prompt Generation**: Creating prompts combining the visual and tracking data with questions
3. **LLM Querying**: Submitting prompts to locally-hosted multimodal LLMs
4. **Result Storage**: Saving responses and generating a structured dataset (parquet)

## Files in this Directory

- `generate_prompts.py`: Script for generating prompts from tracking and detection data
- `query_local_llm.py`: Script for querying a local LLM with generated prompts
- `process_and_save_results.py`: Combined script that processes videos, generates prompts, queries LLMs, and saves results
- `system_prompt.txt`: System prompt for the LLM, providing context for the task
- `sample_questions.json`: Sample questions to use when metadata questions aren't available

## End-to-End Processing with `process_and_save_results.py`

The `process_and_save_results.py` script automates the entire pipeline in one step, from generating prompts to saving structured results.

### Basic Usage

```bash
python src/LLM_prompt_generation/process_and_save_results.py \
  --vis_dir dataset/tracking_visualizations \
  --tracking_dir dataset/tracking_results \
  --yolo_dir dataset/yolov10_results \
  --metadata_file dataset/metadata.parquet \
  --output_dir output \
  --format ollama \
  --model_name llava
```

### Command-Line Arguments

| Argument               | Description                                              | Default                                         |
| ---------------------- | -------------------------------------------------------- | ----------------------------------------------- |
| `--vis_dir`            | Directory containing tracking visualization images       | dataset/tracking_visualizations                 |
| `--tracking_dir`       | Directory containing tracking result JSON files          | dataset/tracking_results                        |
| `--yolo_dir`           | Directory containing YOLOv10 detection result JSON files | dataset/yolov10_results                         |
| `--system_prompt_file` | File containing the system prompt                        | src/LLM_prompt_generation/system_prompt.txt     |
| `--questions_file`     | JSON file with sample questions (fallback)               | src/LLM_prompt_generation/sample_questions.json |
| `--metadata_file`      | Parquet file with video metadata and questions           | dataset/metadata.parquet                        |
| `--output_dir`         | Directory to save prompts, responses, and results        | output                                          |
| `--num_questions`      | Number of questions per prompt                           | 3                                               |
| `--sample_fraction`    | Fraction of YOLOv10 detection images to include          | 0.2 (1/5th)                                     |
| `--format`             | LLM API format (ollama or lmstudio)                      | ollama                                          |
| `--model_name`         | Model name for Ollama                                    | llava                                           |
| `--api_url`            | URL of the LLM API                                       | (depends on format)                             |
| `--videos`             | Specific video IDs to process (optional)                 | (processes all available)                       |
| `--timeout`            | Timeout for LLM API calls in seconds                     | 120                                             |
| `--delay`              | Delay between API calls in seconds                       | 1.0                                             |
| `--save_prompts`       | Save generated prompts to disk                           | False                                           |
| `--resume`             | Resume from previous run, skipping processed videos      | False                                           |

### Data Sources Used

The script uses the following data for each video:

1. **Tracking data**: The `tracking_summary.json` file from each video's folder in `tracking_results`
2. **Tracking visualizations**: All PNG and JPG visualization files from each video's folder in `tracking_visualizations` (excluding MP4 files)
3. **YOLOv10 detection images**: A subset (1/5th by default) of detection images from each video's `detections` folder in `yolov10_results`
4. **Questions**: Questions from the metadata file matching the video ID

### Output Structure

The script creates a structured output directory:

```
output/
├── prompts/                 # (only if --save_prompts is used)
│   ├── video1_prompt.json
│   ├── video2_prompt.json
│   └── ...
├── responses/
│   ├── video1_response.json
│   ├── video2_response.json
│   └── ...
├── results.parquet          # Combined results in parquet format
└── results.csv              # CSV version for easier inspection
```

### Example Commands

#### Processing all videos with metadata questions:

```bash
python src/LLM_prompt_generation/process_and_save_results.py \
  --output_dir output/llm_analysis
```

#### Processing specific videos:

```bash
python src/LLM_prompt_generation/process_and_save_results.py \
  --videos sj81PWrerDk ZQQ6XGuLtyU \
  --output_dir output/specific_videos
```

#### Resuming a partially completed run:

```bash
python src/LLM_prompt_generation/process_and_save_results.py \
  --output_dir output/llm_analysis \
  --resume
```

#### Using LM Studio instead of Ollama:

```bash
python src/LLM_prompt_generation/process_and_save_results.py \
  --format lmstudio \
  --output_dir output/lmstudio_results
```

## Using Individual Scripts

If you prefer to run the pipeline in separate steps, you can use:

1. **Generate prompts**: Use `generate_prompts.py` to create prompts from the data
2. **Query LLM**: Use `query_local_llm.py` to submit the prompts to the LLM

See the `local_llm_README.md` for details on using these individual scripts.

## Requirements

- Python 3.8+
- Required packages:
  - pandas
  - requests
  - tqdm
  - Pillow (PIL)
  - pyarrow (for parquet)
- A locally running instance of either:
  - [Ollama](https://ollama.ai/) with a multimodal model (like LLaVA)
  - [LM Studio](https://lmstudio.ai/) with a multimodal model

## Notes

- Make sure your locally-hosted LLM is running before executing the script
- The script automatically selects videos that have data in all required directories
- For heavy processing, consider using a GPU for faster LLM inference
- If the process fails, you can resume from where it left off using the `--resume` flag
- You can run the script for specific videos using the `--videos` argument
