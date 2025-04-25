#!/usr/bin/env python3
"""
Script to process videos, generate prompts, query the LLM, and save results as parquet.
This script combines the functionality of generate_prompts.py and query_local_llm.py
and adds the ability to save results as a parquet file.
"""

import os
import json
import base64
import time
import argparse
import pandas as pd
import requests
import math
import random
from pathlib import Path
from tqdm import tqdm


def load_metadata_questions(metadata_path, video_id=None):
    """Load questions from the metadata parquet file."""
    try:
        df = pd.read_parquet(metadata_path)
        if video_id:
            video_questions = df[df['video_id']
                                 == video_id]['question'].tolist()
            return video_questions
        else:
            # Return the full dataframe for later processing
            return df
    except Exception as e:
        print(f"Error loading metadata: {e}")
        return [] if video_id else pd.DataFrame()


def load_tracking_results(file_path):
    """Load tracking results from a JSON file."""
    with open(file_path, 'r') as f:
        tracking_data = json.load(f)
    return tracking_data


def load_yolo_results(file_path):
    """Load YOLOv10 detection results from a JSON file."""
    with open(file_path, 'r') as f:
        yolo_data = json.load(f)
    return yolo_data


def encode_image_to_base64(image_path):
    """Encode an image file to base64 for inclusion in the prompt."""
    with open(image_path, 'rb') as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def get_tracking_visualizations(vis_dir, video_id):
    """Get all visualization images (excluding MP4 files) for a video."""
    video_vis_dir = os.path.join(vis_dir, video_id)
    if not os.path.exists(video_vis_dir):
        return []

    # Get all visualization files, excluding .mp4 files
    vis_files = [os.path.join(video_vis_dir, f) for f in os.listdir(video_vis_dir)
                 if f.endswith(('.png', '.jpg', '.jpeg')) and not f.endswith('.mp4')]

    return vis_files


def get_yolo_detection_samples(yolo_dir, video_id, sample_fraction=0.2):
    """Get a subset (1/5th) of YOLOv10 detection images, evenly spaced."""
    detection_dir = os.path.join(yolo_dir, video_id, "detections")
    if not os.path.exists(detection_dir):
        return []

    # Get all detection image files
    detection_files = [f for f in os.listdir(detection_dir)
                       if f.startswith(f"{video_id}_detection_") and f.endswith(('.jpg', '.jpeg', '.png'))]

    # Sort by frame number
    detection_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))

    # Calculate how many images to sample
    total_images = len(detection_files)
    sample_count = max(1, math.ceil(total_images * sample_fraction))

    # Select evenly spaced images
    if sample_count >= total_images:
        selected_detections = detection_files
    else:
        # Evenly sample frames across all detections
        step = total_images / sample_count
        indices = [int(i * step) for i in range(sample_count)]
        selected_detections = [detection_files[i] for i in indices]

    # Return full paths
    return [os.path.join(detection_dir, f) for f in selected_detections]


def generate_prompt(vis_paths, detection_paths, tracking_data, yolo_data, questions, num_questions=3, format_type="ollama"):
    """Generate a complete prompt for the LLM including visualizations, detection images, tracking data, and questions."""
    # Default system prompt embedded in the code
    system_prompt = """You are a specialized video scene analyst tasked with answering questions about what happens in short videos. You will analyze:

1. TRACKING DATA: Contains information about objects in the scene including:
   - Object IDs and classifications
   - Duration objects appear in the video
   - Position and movement trajectories
   - Interactions between different objects

2. VISUAL EVIDENCE: You'll receive sample frames from the video, including:
   - Tracking visualizations showing object paths and movements over time
   - Detection images highlighting identified objects with bounding boxes

When answering questions about the video content:

- Integrate information from both tracking data and visual evidence
- Reference specific evidence from the tracking data (object IDs, trajectories, timings)
- Describe what you observe in the sample images (object positions, actions, interactions)
- Identify key events and timeline of actions in the scene
- Provide concrete, data-supported observations rather than speculations
- Be precise about quantities, positions, and object classifications
- Consider temporal aspects - the sequence of events that occurred
- Focus on the most relevant information to directly answer each question

If tracking data and visual evidence appear inconsistent, acknowledge the discrepancy and explain your reasoning for preferring one source over another."""

    # Encode the visualization images to base64
    encoded_vis = []
    for vis_path in vis_paths:
        try:
            encoded_vis.append({
                "path": vis_path,
                "base64": encode_image_to_base64(vis_path),
                "type": "visualization"
            })
        except Exception as e:
            print(f"Error encoding visualization {vis_path}: {e}")

    # Encode the detection images to base64
    encoded_detections = []
    for detection_path in detection_paths:
        try:
            encoded_detections.append({
                "path": detection_path,
                "base64": encode_image_to_base64(detection_path),
                "type": "detection"
            })
        except Exception as e:
            print(f"Error encoding detection {detection_path}: {e}")

    # Combine all images with their type
    all_images = encoded_vis + encoded_detections

    if not all_images:
        print("No images could be encoded. Cannot generate prompt.")
        return None

    # Select a random subset of questions if we have more than requested
    if len(questions) > num_questions:
        selected_questions = random.sample(questions, num_questions)
    else:
        selected_questions = questions

    # Questions as formatted text
    questions_text = "\n".join(
        [f"{i+1}. {q}" for i, q in enumerate(selected_questions)])

    # Prepare detection data for the prompt
    tracking_info = json.dumps(tracking_data, indent=2)
    yolo_info = json.dumps(yolo_data, indent=2)

    # User message text that includes tracking data, YOLOv10 data, and descriptions of the images
    user_message_text = f"""I'm providing you with tracking data, object detection results, and visualizations for a traffic video.

The visualizations include:
{', '.join([os.path.basename(img['path']) for img in encoded_vis])}

The object detection images include:
{', '.join([os.path.basename(img['path']) for img in encoded_detections])}

Tracking data:
{tracking_info}

YOLOv10 detection data:
{yolo_info}

Please analyze all provided images and data to answer the following questions:
{questions_text}"""

    # Format based on the model type
    if format_type == "anthropic":
        # Claude-style prompt
        content_items = []

        # Add all images first
        for img in all_images:
            content_items.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": img["base64"]
                }
            })

        # Add the text after all images
        content_items.append({
            "type": "text",
            "text": user_message_text
        })

        complete_prompt = [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": content_items
            }
        ]

    elif format_type == "openai":
        # OpenAI-style prompt
        content_items = [{"type": "text", "text": user_message_text}]

        for img in all_images:
            content_items.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{img['base64']}"
                }
            })

        complete_prompt = [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": content_items
            }
        ]

    elif format_type == "ollama":
        # Ollama-style prompt (JSON format)
        images = [
            f"data:image/jpeg;base64,{img['base64']}" for img in all_images]

        complete_prompt = {
            "model": "llava",  # This will be overridden by the actual model name
            "prompt": f"{system_prompt}\n\n{user_message_text}",
            "images": images,
            "stream": False
        }

    elif format_type == "lmstudio":
        # LM Studio style prompt - combine all images into one message
        image_tags = "\n".join(
            [f"<img src='data:image/jpeg;base64,{img['base64']}'>" for img in all_images])

        complete_prompt = {
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": f"{image_tags}\n\n{user_message_text}"
                }
            ],
            "temperature": 0.7,
            "max_tokens": 1024,
            "stream": False
        }

    else:
        # Generic format for other local LLMs
        complete_prompt = {
            "system_prompt": system_prompt,
            "user_message": user_message_text,
            "image_base64": [img["base64"] for img in all_images],
            "questions": selected_questions
        }

    return {
        "prompt": complete_prompt,
        "format_type": format_type,
        "questions": selected_questions,
        "visualization_paths": vis_paths,
        "detection_paths": detection_paths,
        "tracking_data": tracking_data,
        "yolo_data": yolo_data
    }


def query_ollama(prompt_data, model_name="llava", api_url="http://localhost:11434/api/generate", timeout=120):
    """Query an Ollama model with the given prompt data."""
    # Update the model name in the prompt
    prompt_data["prompt"]["model"] = model_name

    # Make the API request
    try:
        response = requests.post(
            api_url, json=prompt_data["prompt"], timeout=timeout)

        if response.status_code == 200:
            # Parse the response - Ollama returns each token as a separate JSON object
            full_response = ""
            for line in response.text.strip().split('\n'):
                try:
                    token_data = json.loads(line)
                    if "response" in token_data:
                        full_response += token_data["response"]
                except json.JSONDecodeError:
                    continue

            return {
                "success": True,
                "response": full_response
            }
        else:
            return {
                "success": False,
                "error": f"API error: {response.status_code} - {response.text}"
            }
    except requests.exceptions.Timeout:
        return {
            "success": False,
            "error": f"Request timed out after {timeout} seconds"
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Error querying Ollama: {str(e)}"
        }


def query_lmstudio(prompt_data, api_url="http://localhost:1234/v1/chat/completions", timeout=120):
    """Query an LM Studio model with the given prompt data."""
    # Make the API request
    try:
        response = requests.post(
            api_url, json=prompt_data["prompt"], timeout=timeout)

        if response.status_code == 200:
            response_data = response.json()
            if "choices" in response_data and len(response_data["choices"]) > 0:
                return {
                    "success": True,
                    "response": response_data["choices"][0]["message"]["content"]
                }
            else:
                return {
                    "success": False,
                    "error": "Invalid response format from LM Studio"
                }
        else:
            return {
                "success": False,
                "error": f"API error: {response.status_code} - {response.text}"
            }
    except requests.exceptions.Timeout:
        return {
            "success": False,
            "error": f"Request timed out after {timeout} seconds"
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Error querying LM Studio: {str(e)}"
        }


def process_video(video_id, metadata_df, args):
    """Process a single video: generate prompt, query LLM, and return result."""
    print(f"Processing video: {video_id}")

    # Get visualization images for this video (excluding mp4 files)
    vis_paths = get_tracking_visualizations(args.vis_dir, video_id)
    if not vis_paths:
        print(f"No visualization images found for video {video_id}. Skipping.")
        return None

    # Get a sample of YOLO detection images
    detection_paths = get_yolo_detection_samples(
        args.yolo_dir, video_id, args.sample_fraction)
    if not detection_paths:
        print(
            f"No YOLOv10 detection images found for video {video_id}. Skipping.")
        return None

    # Get tracking data path
    tracking_path = os.path.join(
        args.tracking_dir, video_id, "tracking_summary.json")
    if not os.path.exists(tracking_path):
        print(f"No tracking data found for video {video_id}. Skipping.")
        return None

    # Get YOLOv10 data path
    yolo_path = os.path.join(args.yolo_dir, video_id,
                             "json_detections", f"{video_id}_detections.json")
    if not os.path.exists(yolo_path):
        print(f"No YOLOv10 data found for video {video_id}. Skipping.")
        return None

    # Load data
    tracking_data = load_tracking_results(tracking_path)
    yolo_data = load_yolo_results(yolo_path)

    # Get question records from metadata for this video
    video_questions_df = metadata_df[metadata_df['video_id'] == video_id]

    if video_questions_df.empty:
        print(
            f"No questions found in metadata for video {video_id}. Skipping video.")
        return None
    else:
        questions = video_questions_df['question'].tolist()

    # Generate prompt
    prompt_data = generate_prompt(
        vis_paths,
        detection_paths,
        tracking_data,
        yolo_data,
        questions,
        args.num_questions,
        args.format
    )

    if not prompt_data:
        print(f"Failed to generate prompt for video {video_id}. Skipping.")
        return None

    # Save prompt to output directory if needed
    if args.save_prompts:
        prompts_dir = os.path.join(args.output_dir, "prompts")
        os.makedirs(prompts_dir, exist_ok=True)
        output_file = os.path.join(prompts_dir, f"{video_id}_prompt.json")

        # Remove the base64 data for saving to reduce file size
        save_data = prompt_data.copy()

        # Replace base64 data with file references
        save_data["visualization_references"] = vis_paths
        save_data["detection_references"] = detection_paths

        if args.format == 'ollama':
            save_data["prompt"]["images"] = [
                "[BASE64_IMAGE_DATA_REMOVED]"] * len(vis_paths + detection_paths)
        elif args.format == 'lmstudio':
            # Replace each <img> tag with a placeholder
            content = save_data["prompt"]["messages"][1]["content"]
            for i in range(len(vis_paths) + len(detection_paths)):
                content = content.replace(
                    f"<img src='data:image/jpeg;base64,{prompt_data['prompt']['messages'][1]['content'].split('base64,')[i+1].split('>')[0]}'>",
                    f"<img src='[BASE64_IMAGE_DATA_REMOVED_{i}]'>"
                )
            save_data["prompt"]["messages"][1]["content"] = content

        with open(output_file, 'w') as f:
            json.dump(save_data, f, indent=2)

        print(f"Saved prompt to {output_file}")

    # Query the LLM
    print(f"Querying LLM for video {video_id}...")
    if args.format == 'ollama':
        response = query_ollama(
            prompt_data, args.model_name, args.api_url, args.timeout)
    elif args.format == 'lmstudio':
        response = query_lmstudio(prompt_data, args.api_url, args.timeout)
    else:
        response = {
            "success": False,
            "error": f"Unsupported format: {args.format}"
        }

    # Save response
    responses_dir = os.path.join(args.output_dir, "responses")
    os.makedirs(responses_dir, exist_ok=True)
    response_file = os.path.join(responses_dir, f"{video_id}_response.json")

    response_data = {
        "video_id": video_id,
        "questions": prompt_data["questions"],
        "success": response["success"],
        "response": response.get("response", None),
        "error": response.get("error", None),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    with open(response_file, 'w') as f:
        json.dump(response_data, f, indent=2)

    print(f"Saved response to {response_file}")

    # Prepare results for the dataframe
    results = []

    # If the response was successful, we need to pair each question with the response
    if response["success"]:
        for qid, question in enumerate(prompt_data["questions"]):
            results.append({
                "video_id": video_id,
                "question": question,
                "question_index": qid,
                "answer": response["response"],
                "success": True,
                "error": None,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            })
    else:
        # If there was an error, add one row per question with the error
        for qid, question in enumerate(prompt_data["questions"]):
            results.append({
                "video_id": video_id,
                "question": question,
                "question_index": qid,
                "answer": None,
                "success": False,
                "error": response.get("error", "Unknown error"),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            })

    return results


def main():
    parser = argparse.ArgumentParser(
        description='Process videos, generate prompts, query the LLM, and save results as parquet.')
    parser.add_argument('--vis_dir', type=str, default='dataset/tracking_visualizations',
                        help='Directory containing tracking visualization images')
    parser.add_argument('--tracking_dir', type=str, default='dataset/tracking_results',
                        help='Directory containing tracking result JSON files')
    parser.add_argument('--yolo_dir', type=str, default='dataset/yolov10_results',
                        help='Directory containing YOLOv10 detection result JSON files')
    parser.add_argument('--metadata_file', type=str, default='dataset/metadata.parquet',
                        help='Parquet file containing video metadata with questions')
    parser.add_argument('--output_dir', type=str, default='output',
                        help='Directory to save the generated prompts, responses, and results')
    parser.add_argument('--num_questions', type=int, default=3,
                        help='Number of questions to include in each prompt (default: 3)')
    parser.add_argument('--sample_fraction', type=float, default=0.2,
                        help='Fraction of YOLOv10 detection images to include (default: 0.2 or 1/5th)')
    parser.add_argument('--format', type=str, default='ollama', choices=['ollama', 'lmstudio'],
                        help='Format type for the prompt (default: ollama)')
    parser.add_argument('--model_name', type=str, default='llava',
                        help='Model name to use (for Ollama only, e.g., llava)')
    parser.add_argument('--api_url', type=str, default=None,
                        help='URL of the LLM API (default depends on format)')
    parser.add_argument('--videos', type=str, nargs='+',
                        help='Specific video IDs to process (optional)')
    parser.add_argument('--timeout', type=int, default=120,
                        help='Timeout for LLM API calls in seconds (default: 120)')
    parser.add_argument('--delay', type=float, default=1.0,
                        help='Delay between API calls in seconds (default: 1.0)')
    parser.add_argument('--save_prompts', action='store_true',
                        help='Save generated prompts to disk (without base64 data)')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from previous run, skipping videos with existing responses')

    args = parser.parse_args()

    # Set default API URLs if not provided
    if not args.api_url:
        if args.format == 'ollama':
            args.api_url = "http://localhost:11434/api/generate"
        else:  # lmstudio
            args.api_url = "http://localhost:1234/v1/chat/completions"

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Load metadata
    metadata_df = load_metadata_questions(args.metadata_file)
    if metadata_df.empty:
        print("Error: Could not load metadata. Exiting.")
        return

    # Get list of videos to process
    if args.videos:
        video_ids = args.videos
    else:
        # Use all videos that have data in all required directories
        tracking_videos = set(os.listdir(args.tracking_dir))
        yolo_videos = set(os.listdir(args.yolo_dir))
        vis_videos = set(os.listdir(args.vis_dir))

        # Find intersection of videos present in all directories
        video_ids = list(tracking_videos.intersection(yolo_videos, vis_videos))

    print(f"Found {len(video_ids)} videos to process")

    # If resuming, check which videos already have responses
    if args.resume:
        responses_dir = os.path.join(args.output_dir, "responses")
        if os.path.exists(responses_dir):
            existing_responses = [f.replace("_response.json", "") for f in os.listdir(responses_dir)
                                  if f.endswith("_response.json")]
            print(f"Found {len(existing_responses)} existing responses")
            # Filter out videos that already have responses
            video_ids = [
                vid for vid in video_ids if vid not in existing_responses]
            print(f"Will process {len(video_ids)} remaining videos")

    # Process each video and collect results
    all_results = []

    for video_id in tqdm(video_ids, desc="Processing videos"):
        results = process_video(video_id, metadata_df, args)
        if results:
            all_results.extend(results)

        # Add delay between videos
        if video_id != video_ids[-1]:  # Skip delay after the last video
            time.sleep(args.delay)

    # Convert results to dataframe
    if all_results:
        results_df = pd.DataFrame(all_results)

        # Save results as parquet
        parquet_path = os.path.join(args.output_dir, "results.parquet")

        # If resuming and the parquet file exists, append to it
        if args.resume and os.path.exists(parquet_path):
            existing_df = pd.read_parquet(parquet_path)
            # Combine with new results
            combined_df = pd.concat(
                [existing_df, results_df], ignore_index=True)
            combined_df.to_parquet(parquet_path, index=False)
            print(f"Appended new results to {parquet_path}")
        else:
            # Save as new file
            results_df.to_parquet(parquet_path, index=False)
            print(f"Saved results to {parquet_path}")

        # Also save as CSV for easier inspection
        csv_path = os.path.join(args.output_dir, "results.csv")
        if args.resume and os.path.exists(parquet_path):
            combined_df.to_csv(csv_path, index=False)
        else:
            results_df.to_csv(csv_path, index=False)

        print(
            f"Processed {len(all_results)} questions across {len(set([r['video_id'] for r in all_results]))} videos")
    else:
        print("No results were generated.")


if __name__ == "__main__":
    main()
