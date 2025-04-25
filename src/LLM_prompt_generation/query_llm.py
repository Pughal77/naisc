#!/usr/bin/env python3
"""
Script to query LLMs directly using Hugging Face transformers.
This script supports both original prompts and lightweight prompts with tracking data only.
"""

import os
import json
import argparse
import base64
import time
from pathlib import Path
from tqdm import tqdm
import torch
from PIL import Image
import io
import numpy as np

# Import Hugging Face libraries
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer
from transformers import pipeline


def load_prompt(prompt_file):
    """Load a saved prompt file."""
    with open(prompt_file, 'r') as f:
        return json.load(f)


def decode_base64_to_image(base64_string):
    """Convert base64 string to a PIL Image."""
    if "base64," in base64_string:
        # Extract the actual base64 part if it includes the data URL prefix
        base64_string = base64_string.split("base64,")[1]

    image_data = base64.b64decode(base64_string)
    return Image.open(io.BytesIO(image_data))


def load_image_from_reference(image_path):
    """Load an image from a file path."""
    return Image.open(image_path).convert("RGB")


def extract_text_from_prompt(prompt_data):
    """Extract the text prompt from the prompt data."""
    if prompt_data["format_type"] == "ollama":
        return prompt_data["prompt"]["prompt"]
    elif prompt_data["format_type"] == "lmstudio":
        # Find the user message (usually the second message)
        for message in prompt_data["prompt"]["messages"]:
            if message["role"] == "user":
                # Extract text part (remove image tags if present)
                content = message["content"]
                if isinstance(content, str) and "<img src=" in content:
                    # Extract text before and after the image tag
                    parts = content.split("<img src=")
                    if len(parts) > 1:
                        after_img = parts[1].split(">", 1)
                        if len(after_img) > 1:
                            content = parts[0] + after_img[1]
                return content

    # Fallback - try to find a question field
    if "question" in prompt_data:
        return prompt_data["question"]
    elif "questions" in prompt_data:
        return " ".join(prompt_data["questions"])

    # Last resort
    return "Please analyze this image and tracking data."


def query_model(model, processor, prompt_data, device):
    """Query a model with the given prompt data."""
    try:
        # Extract text from the prompt
        text_prompt = extract_text_from_prompt(prompt_data)

        # Get image if available
        image = None
        if "image_reference" in prompt_data:
            image = load_image_from_reference(prompt_data["image_reference"])
        elif prompt_data["format_type"] == "ollama" and "images" in prompt_data["prompt"]:
            if prompt_data["prompt"]["images"]:
                image = decode_base64_to_image(
                    prompt_data["prompt"]["images"][0])

        # Process inputs
        if image is not None:
            # This is for multimodal models
            inputs = processor(text=text_prompt, images=image,
                               return_tensors="pt").to(device)

            # Generate response with autoregressive generation
            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                )

            # Decode the generated ids
            generated_text = processor.batch_decode(
                output_ids, skip_special_tokens=True)[0]
            response_text = generated_text.split(text_prompt)[-1].strip()
        else:
            # Text-only model
            inputs = processor(text_prompt, return_tensors="pt").to(device)

            with torch.no_grad():
                output_ids = model.generate(
                    inputs["input_ids"],
                    max_new_tokens=512,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                )

            response_text = processor.decode(
                output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

        return {
            "success": True,
            "response": response_text
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Model error: {str(e)}"
        }


def save_response(response_data, output_file):
    """Save the response to a file."""
    with open(output_file, 'w') as f:
        json.dump(response_data, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description='Query LLMs directly using Hugging Face transformers.')
    parser.add_argument('--prompt_dir', type=str, required=True,
                        help='Directory containing generated prompt files')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save responses')
    parser.add_argument('--model_name', type=str, default='mistralai/Mistral-7B-Instruct-v0.2',
                        help='Hugging Face model name or path')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to run model on (cuda, cpu)')
    parser.add_argument('--delay', type=float, default=0.5,
                        help='Delay between model calls in seconds (default: 0.5)')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size for processing prompts (default: 1)')
    parser.add_argument('--precision', type=str, choices=['fp16', 'fp32', 'int8', 'int4'], default='fp16',
                        help='Model precision (default: fp16)')

    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Determine correct device
    device = torch.device(args.device)
    print(f"Using device: {device}")

    # Load model and processor
    print(f"Loading model: {args.model_name}")

    # Set precision options
    dtype = torch.float16 if args.precision == 'fp16' else torch.float32
    quantization = None
    if args.precision == 'int8':
        quantization = "int8"
    elif args.precision == 'int4':
        quantization = "int4"

    # Try to load as a multimodal model first
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            torch_dtype=dtype,
            device_map=args.device,
            quantization_config=quantization
        )
        processor = AutoProcessor.from_pretrained(args.model_name)
        print("Loaded model as multimodal model")
    except Exception as e:
        print(
            f"Could not load as multimodal model, trying as text-only model: {e}")
        try:
            model = AutoModelForCausalLM.from_pretrained(
                args.model_name,
                torch_dtype=dtype,
                device_map=args.device,
                quantization_config=quantization
            )
            processor = AutoTokenizer.from_pretrained(args.model_name)
            print("Loaded model as text-only model")
        except Exception as e:
            print(f"Failed to load model: {e}")
            return

    # Get list of prompt files - support both original and lightweight naming formats
    prompt_dir = Path(args.prompt_dir)
    prompt_files = list(prompt_dir.glob("*_prompt.json")) + \
        list(prompt_dir.glob("*_lightweight_prompt.json"))

    if not prompt_files:
        print(f"No prompt files found in {args.prompt_dir}")
        return

    # Process each prompt file
    for prompt_file in tqdm(prompt_files, desc="Querying model"):
        # Load the prompt
        prompt_data = load_prompt(str(prompt_file))

        # Query the model
        response = query_model(model, processor, prompt_data, device)

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

        # Add delay to avoid overloading the GPU
        time.sleep(args.delay)

    print(f"Processing complete. Responses saved to {args.output_dir}")


if __name__ == "__main__":
    main()
