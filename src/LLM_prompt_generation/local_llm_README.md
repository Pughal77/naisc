# Local LLM Video Scene Analysis

This guide explains how to use locally-hosted multimodal Large Language Models (LLMs) with our video scene analysis tools. The tools process tracking data, visualizations, and object detection results to analyze content in short videos.

## Prerequisites

- Python 3.8+
- Required packages:
  - pandas
  - requests
  - tqdm
  - Pillow (PIL)
  - pyarrow (for parquet)
- A locally running instance of either:
  - [Ollama](https://ollama.ai/) with a multimodal model like LLaVA
  - [LM Studio](https://lmstudio.ai/) with a multimodal model

## Option 1: End-to-End Processing (Recommended)

The `process_and_save_results.py` script combines prompt generation, LLM querying, and results saving in one step:

```bash
python src/LLM_prompt_generation/process_and_save_results.py \
  --vis_dir dataset/tracking_visualizations \
  --tracking_dir dataset/tracking_results \
  --yolo_dir dataset/yolov10_results \
  --metadata_file dataset/metadata.parquet \
  --output_dir output/llm_analysis \
  --format ollama \
  --model_name llava
```

### Key Features

- Processes all videos in a single command
- Uses questions from the metadata.parquet file
- Queries the LLM and saves both JSON responses and structured parquet/CSV files
- Supports resuming interrupted processing with `--resume`
- Handles all data collection, prompt formatting, and result saving

### Main Arguments

| Argument            | Description                                        | Default                         |
| ------------------- | -------------------------------------------------- | ------------------------------- |
| `--vis_dir`         | Directory with tracking visualization images       | dataset/tracking_visualizations |
| `--tracking_dir`    | Directory with tracking result JSON files          | dataset/tracking_results        |
| `--yolo_dir`        | Directory with YOLOv10 detection result JSON files | dataset/yolov10_results         |
| `--metadata_file`   | Parquet file with video metadata and questions     | dataset/metadata.parquet        |
| `--output_dir`      | Directory to save results                          | output                          |
| `--format`          | Format type (`ollama` or `lmstudio`)               | ollama                          |
| `--model_name`      | Model name (for Ollama only, e.g., `llava`)        | llava                           |
| `--sample_fraction` | Fraction of YOLOv10 detection images to include    | 0.2 (1/5th)                     |
| `--videos`          | Specific video IDs to process (optional)           | (processes all available)       |
| `--timeout`         | Timeout for LLM API calls in seconds               | 120                             |
| `--delay`           | Delay between API calls in seconds                 | 1.0                             |
| `--save_prompts`    | Save generated prompts to disk                     | False                           |
| `--resume`          | Resume from previous run                           | False                           |

See `python src/LLM_prompt_generation/process_and_save_results.py --help` for all options.

## Option 2: Step-by-Step Processing

If you prefer more control, you can run the pipeline in separate steps.

### Step 1: Generate Prompts

```bash
python src/LLM_prompt_generation/generate_prompts.py \
  --vis_dir dataset/tracking_visualizations \
  --tracking_dir dataset/tracking_results \
  --yolo_dir dataset/yolov10_results \
  --metadata_file dataset/metadata.parquet \
  --output_dir output/prompts \
  --format ollama
```

#### Key Arguments:

- `--vis_dir`: Directory with tracking visualization images
- `--tracking_dir`: Directory with tracking result JSON files
- `--yolo_dir`: Directory with YOLOv10 detection result JSON files
- `--metadata_file`: Parquet file with video metadata and questions
- `--output_dir`: Output directory for prompts
- `--num_questions`: Questions per prompt (default: 3)
- `--sample_fraction`: Fraction of detection images to include (default: 0.2)
- `--format`: Format type (`ollama`, `lmstudio`, `anthropic`, `openai`, `generic`)
- `--videos`: Specific video IDs to process (optional)

### Step 2: Query Local LLM

```bash
python src/LLM_prompt_generation/query_local_llm.py \
  --prompt_dir output/prompts \
  --output_dir output/responses \
  --model_type ollama \
  --model_name llava
```

#### Key Arguments:

- `--prompt_dir`: Directory with generated prompt files
- `--output_dir`: Directory to save responses
- `--model_type`: LLM type (`ollama` or `lmstudio`)
- `--model_name`: Model name (for Ollama only)
- `--api_url`: API URL (defaults: Ollama=localhost:11434, LM Studio=localhost:1234)
- `--delay`: Delay between API calls in seconds (default: 1.0)

## Data Sources

Both approaches use the following data for each video:

1. **Tracking data**: `tracking_summary.json` from each video folder in `tracking_results`
2. **Tracking visualizations**: PNG/JPG files from each video folder in `tracking_visualizations`
3. **YOLOv10 detection images**: Sample of images from each video's `detections` folder in `yolov10_results`
4. **Questions**: Video-specific questions from `metadata.parquet`

## Default System Prompt

The scripts use a default system prompt embedded in the code that instructs the model to:

- Analyze tracking data and visual evidence from videos
- Identify objects, movements, and interactions in the scene
- Combine information from different data sources to answer questions
- Focus on observable facts and provide evidence-based answers

## Setting Up Local LLMs

### Ollama Setup

1. Download and install [Ollama](https://ollama.ai/)
2. Pull a multimodal model:
   ```bash
   ollama pull llava
   ```
3. Start the server:
   ```bash
   ollama serve
   ```

### LM Studio Setup

1. Download and install [LM Studio](https://lmstudio.ai/)
2. Download a multimodal model (LLaVA, Bakllava, etc.)
3. Start the local server via the UI

## Example Workflows

### End-to-End (Recommended)

Process all videos:

```bash
python src/LLM_prompt_generation/process_and_save_results.py \
  --output_dir output/llm_analysis \
  --format ollama \
  --model_name llava
```

Process specific videos:

```bash
python src/LLM_prompt_generation/process_and_save_results.py \
  --videos sj81PWrerDk ZQQ6XGuLtyU \
  --output_dir output/specific_videos \
  --format ollama \
  --model_name llava
```

Resume interrupted processing:

```bash
python src/LLM_prompt_generation/process_and_save_results.py \
  --output_dir output/llm_analysis \
  --format ollama \
  --model_name llava \
  --resume
```

### Two-Step Alternative

1. Generate prompts:

   ```bash
   python src/LLM_prompt_generation/generate_prompts.py \
     --metadata_file dataset/metadata.parquet \
     --output_dir output/ollama_prompts \
     --format ollama
   ```

2. Query LLM:
   ```bash
   python src/LLM_prompt_generation/query_local_llm.py \
     --prompt_dir output/ollama_prompts \
     --output_dir output/ollama_responses \
     --model_type ollama \
     --model_name llava
   ```

## Output Structure

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
├── results.parquet          # Structured analysis data
└── results.csv              # CSV version for easier viewing
```

## Important Notes

- Videos without questions in the metadata file will be skipped
- The scripts automatically match videos across the tracking, visualization, and YOLOv10 directories
- Only videos present in all three directories will be processed
- No external files for system prompts or sample questions are needed

## Troubleshooting

- **API connection error**: Ensure your LLM server is running and the API URL is correct
- **Slow responses**: Reduce the delay between queries or the number of images
- **Out of memory errors**: Try a different model, reduce sample fraction, or resize images
- **Missing data**: The script will skip videos missing data in any required directory
- **Missing questions**: Videos without questions in metadata will be skipped
- **Interrupted processing**: Use `--resume` to continue from where you left off
