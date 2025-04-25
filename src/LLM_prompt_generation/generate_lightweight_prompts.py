#!/usr/bin/env python3
"""
Script to generate lightweight prompts for LLMs using only tracking data.
This version creates much smaller prompts by excluding image data and generates 
one prompt per question.
"""

import os
import json
import argparse
import pandas as pd


def load_metadata_questions(metadata_path, video_id):
    """Load questions for a specific video from the metadata parquet file."""
    try:
        df = pd.read_parquet(metadata_path)
        # Get rows that match the video_id
        video_questions = df[df['video_id'] == video_id]
        # Return the dataframe rows so we can access question and question_id
        return video_questions
    except Exception as e:
        print(
            f"Error loading questions from metadata for video {video_id}: {e}")
        return pd.DataFrame()


def load_tracking_results(file_path):
    """Load tracking results from a JSON file."""
    with open(file_path, 'r') as f:
        tracking_data = json.load(f)
    return tracking_data


def generate_lightweight_prompt(tracking_data, question, format_type="lmstudio"):
    """Generate a prompt using only tracking data without images for a single question."""
    # Default system prompt
    system_prompt = """You are a specialized video scene analyst tasked with answering questions based on tracking data.

The tracking data contains information about objects in the scene including:
- Object IDs and classifications
- Duration objects appear in the video
- Position and movement trajectories
- Interactions between different objects

When answering questions:
- Reference specific evidence from the tracking data (object IDs, trajectories, timings)
- Identify key events and timeline of actions in the scene
- Provide concrete, data-supported observations based on the tracking information
- Be precise about quantities, positions, and object classifications
- Consider temporal aspects - the sequence of events that occurred
- Focus on the most relevant information to directly answer the question

Your answer should be based solely on the tracking data provided."""

    # Prepare tracking data for the prompt
    tracking_info = json.dumps(tracking_data, indent=2)

    # User message text that includes tracking data and a single question
    user_message_text = f"""I'm providing you with tracking data for a video scene.

Tracking data:
{tracking_info}

Please analyze this tracking data to answer the following question:
{question}"""

    # Format based on the model type
    if format_type == "anthropic":
        # Claude-style prompt
        complete_prompt = [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_message_text
            }
        ]

    elif format_type == "openai":
        # OpenAI-style prompt
        complete_prompt = [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_message_text
            }
        ]

    elif format_type == "ollama":
        # Ollama-style prompt (JSON format)
        complete_prompt = {
            "model": "llama3",  # This will be overridden by the actual model name
            "prompt": f"{system_prompt}\n\n{user_message_text}",
            "stream": False
        }

    elif format_type == "lmstudio":
        # LM Studio style prompt
        complete_prompt = {
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": user_message_text
                }
            ],
            "temperature": 0.7,
            "max_tokens": 1024,
            "stream": False
        }

    else:
        # Generic format for other LLMs
        complete_prompt = {
            "system_prompt": system_prompt,
            "user_message": user_message_text,
            "question": question
        }

    return {
        "prompt": complete_prompt,
        "format_type": format_type,
        "question": question,
        "tracking_data": tracking_data
    }


def main():
    parser = argparse.ArgumentParser(
        description='Generate lightweight prompts using only tracking data (no images).')
    parser.add_argument('--tracking_dir', type=str, default='dataset/tracking_results',
                        help='Directory containing tracking result JSON files')
    parser.add_argument('--metadata_file', type=str, default='dataset/metadata.parquet',
                        help='Parquet file containing video metadata with questions')
    parser.add_argument('--output_dir', type=str, default='output/lightweight_prompts',
                        help='Directory to save the generated prompts')
    parser.add_argument('--format', type=str, default='lmstudio',
                        choices=['ollama', 'lmstudio',
                                 'anthropic', 'openai', 'generic'],
                        help='Format type for the prompt (default: ollama)')
    parser.add_argument('--videos', type=str, nargs='+',
                        help='Specific video IDs to process (optional)')

    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Get list of videos to process
    if args.videos:
        video_ids = args.videos
    else:
        # Use all videos that have tracking data
        video_ids = [d for d in os.listdir(args.tracking_dir) if os.path.isdir(
            os.path.join(args.tracking_dir, d))]

    print(f"Found {len(video_ids)} videos to process")
    total_prompts_generated = 0

    # Process each video
    for video_id in video_ids:
        print(f"Processing video: {video_id}")

        # Get tracking data path
        tracking_path = os.path.join(
            args.tracking_dir, video_id, "tracking_summary.json")
        if not os.path.exists(tracking_path):
            print(f"No tracking data found for video {video_id}. Skipping.")
            continue

        # Load tracking data
        tracking_data = load_tracking_results(tracking_path)

        # Get video-specific questions from metadata
        questions_df = load_metadata_questions(args.metadata_file, video_id)
        if questions_df.empty:
            print(
                f"No questions found in metadata for video {video_id}. Skipping.")
            continue

        # Count number of questions for this video
        num_questions = len(questions_df)
        print(f"Found {num_questions} questions for video {video_id}")
        prompts_for_video = 0

        # Generate one prompt per question
        for _, row in questions_df.iterrows():
            question = row['question']
            question_id = row.name if 'question_id' not in row else row['question_id']

            # Generate lightweight prompt for this single question
            prompt_data = generate_lightweight_prompt(
                tracking_data,
                question,
                args.format
            )

            # Create a filename that includes both video_id and question identifier
            output_file = os.path.join(
                args.output_dir, f"{video_id}_q{question_id}_lightweight_prompt.json")

            with open(output_file, 'w') as f:
                json.dump(prompt_data, f, indent=2)

            prompts_for_video += 1

        print(f"Generated {prompts_for_video} prompts for video {video_id}")
        total_prompts_generated += prompts_for_video

    print(
        f"Processed {len(video_ids)} videos. Generated {total_prompts_generated} lightweight prompts (one per question) saved to {args.output_dir}.")


if __name__ == "__main__":
    main()
