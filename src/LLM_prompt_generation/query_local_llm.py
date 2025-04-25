#!/usr/bin/env python3
"""
Script to query locally hosted LLMs (Ollama, LM Studio, etc.) with generated prompts.
This script supports both original prompts and lightweight prompts with tracking data only.
"""

import os
import json
import argparse
import base64
import requests
import time
from pathlib import Path
from tqdm import tqdm


def load_prompt(prompt_file):
    """Load a saved prompt file."""
    with open(prompt_file, 'r') as f:
        return json.load(f)


def restore_image_base64(prompt_data, image_path):
    """Restore the base64 image data to the prompt."""
    with open(image_path, 'rb') as image_file:
        image_base64 = base64.b64encode(image_file.read()).decode('utf-8')

    if prompt_data["format_type"] == "ollama":
        prompt_data["prompt"]["images"] = [
            f"data:image/jpeg;base64,{image_base64}"]
    elif prompt_data["format_type"] == "lmstudio":
        prompt_data["prompt"]["messages"][1]["content"] = prompt_data["prompt"]["messages"][1]["content"].replace(
            "<img src='[BASE64_IMAGE_DATA_REMOVED]'>",
            f"<img src='data:image/jpeg;base64,{image_base64}'>"
        )

    return prompt_data


def query_ollama(prompt_data, model_name="llama3", api_url="http://localhost:11434/api/generate"):
    """Query an Ollama model with the given prompt data."""
    # Update the model name in the prompt
    prompt_data["prompt"]["model"] = model_name

    # Make the API request
    response = requests.post(api_url, json=prompt_data["prompt"])

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


def query_lmstudio(prompt_data, api_url="http://localhost:1234/v1/chat/completions"):
    """Query an LM Studio model with the given prompt data."""
    # Make the API request
    response = requests.post(api_url, json=prompt_data["prompt"])

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


def save_response(response_data, output_file):
    """Save the response to a file."""
    with open(output_file, 'w') as f:
        json.dump(response_data, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description='Query locally hosted LLMs with generated prompts (including lightweight prompts).')
    parser.add_argument('--prompt_dir', type=str, required=True,
                        help='Directory containing generated prompt files')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save responses')
    parser.add_argument('--model_type', type=str, choices=['ollama', 'lmstudio'], default='lmstudio',
                        help='Type of local LLM to query (default: ollama)')
    parser.add_argument('--model_name', type=str, default='mistralai.mistral-small-3.1-24b-base-2503',
                        help='Model name to use (for Ollama, e.g., llama3, mistral, etc.)')
    parser.add_argument('--api_url', type=str,
                        help='URL of the LLM API (default depends on model_type)')
    parser.add_argument('--delay', type=float, default=1.0,
                        help='Delay between API calls in seconds (default: 1.0)')

    args = parser.parse_args()

    # Set default API URLs if not provided
    if not args.api_url:
        if args.model_type == 'ollama':
            args.api_url = "http://localhost:11434/api/generate"
        else:  # lmstudio
            args.api_url = "http://localhost:1234/v1/chat/completions"

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Get list of prompt files - support both original and lightweight naming formats
    prompt_dir = Path(args.prompt_dir)
    prompt_files = list(prompt_dir.glob("*_prompt.json")) + \
        list(prompt_dir.glob("*_lightweight_prompt.json"))

    if not prompt_files:
        print(f"No prompt files found in {args.prompt_dir}")
        return

    # Process each prompt file
    for prompt_file in tqdm(prompt_files, desc="Querying LLM"):
        # Load the prompt
        prompt_data = load_prompt(str(prompt_file))

        # Check if the format type matches the requested model type
        if prompt_data["format_type"] != args.model_type:
            print(
                f"Warning: Prompt format ({prompt_data['format_type']}) doesn't match requested model type ({args.model_type}). Skipping {prompt_file.name}")
            continue

        # Restore the base64 image data if needed
        if "image_reference" in prompt_data:
            prompt_data = restore_image_base64(
                prompt_data, prompt_data["image_reference"])

        # Query the LLM
        if args.model_type == 'ollama':
            response = query_ollama(prompt_data, args.model_name, args.api_url)
        else:  # lmstudio
            response = query_lmstudio(prompt_data, args.api_url)

        # Determine output filename based on input filename pattern
        if "lightweight_prompt.json" in prompt_file.name:
            # For lightweight prompts (videoID_qXXXX_lightweight_prompt.json)
            output_file = Path(
                args.output_dir) / prompt_file.name.replace("_lightweight_prompt.json", "_response.json")
        else:
            # For original prompts (videoID_prompt.json)
            output_file = Path(
                args.output_dir) / prompt_file.name.replace("_prompt.json", "_response.json")

        # Prepare response data based on whether it's a lightweight prompt (single question) or not
        response_data = {
            "prompt_file": str(prompt_file),
            "success": response["success"],
            "response": response["response"] if response["success"] else None,
            "error": response.get("error", None),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }

        # Add either "question" or "questions" key based on the prompt format
        if "question" in prompt_data:
            # Lightweight prompt with a single question
            response_data["question"] = prompt_data["question"]
        elif "questions" in prompt_data:
            # Original prompt with multiple questions
            response_data["questions"] = prompt_data["questions"]

        # Save the response
        save_response(response_data, str(output_file))

        print(f"Processed {prompt_file.name} -> {output_file.name}")

        # Add delay to avoid overloading the LLM
        time.sleep(args.delay)

    print(f"Processing complete. Responses saved to {args.output_dir}")


if __name__ == "__main__":
    main()
