"""
Divergentti 18.2.2025

Version 0.01

Simple command line script for the Coqui-ai TTS https://github.com/coqui-ai/TTS

First install the Coqui-ai to the venv (pyton3.11 venv venv), then activate source venv/bin/activate
Once virtual environment is activated, then clone the repo:
   git clone https://github.com/coqui-ai/TTS.git
   pip install --upgrade pip
   pip install -r requirements.txt
   pip install -e .

Note! Python 3.12.3 is too new, prefer 3.11

If you do not have complier, then:
    sudo apt update
    sudo apt install python3.11-dev python3-pip python3-venv

Test with command:
    tts --list_models

If you prefer CUDA support, keep venv activated and execute:
    pip install torch

Place this script to TTS-folder. Usage of the script:

1. echo "This is piped text" | python speak.py -o output.wav -s p317
2. python speak.py -f your_script.txt -o output.wav -s p317
3. python speak.py -t "This is direct text input" -o output.wav -s p273

"""

from TTS.api import TTS
import sys
import argparse

try:
    # Following are for GPU acceleration (CUDA)
    import torch
    from torch.cuda import is_available as cuda_available, device

    # Check for CUDA availability
    if not cuda_available():
        print("CUDA is not available. Running on CPU instead.")
        gpu_enabled = "false"
    else:
        gpu_enabled = "true"
        print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
except Exception as e:
    print(f"No torch installed: {e}")
    gpu_enabled = "false"

# Set up argument parser for more flexibility
parser = argparse.ArgumentParser(description='Generate TTS audio with VCTK male voice')
parser.add_argument('-t', '--text', help='Text string to convert to speech')
parser.add_argument('-f', '--file', help='Text file to convert to speech')
parser.add_argument('-o', '--output', default='vctk_output.wav', help='Output audio file')
parser.add_argument('-s', '--speaker', default='p317', help='Speaker ID to use')
args = parser.parse_args()

# Initialize TTS with VCTK VITS model
tts = TTS(model_name="tts_models/en/vctk/vits")

# Determine the text source
if args.file:
    try:
        with open(args.file, 'r', encoding='utf-8') as f:
            text = f.read()
    except Exception as e:
        print(f"Error reading file: {e}")
        sys.exit(1)
elif args.text:
    text = args.text
else:
    # If no text is provided, read from stdin (for piping)
    if not sys.stdin.isatty():  # Check if data is being piped in
        text = sys.stdin.read()
    else:
        print("No text provided. Please use -t, -f, or pipe text to the script.")
        sys.exit(1)

# Generate TTS with specified speaker
try:
    tts.tts_to_file(
        text=text,
        file_path=args.output,
        speaker=args.speaker,
        gpu=gpu_enabled
    )
    print(f"Generated voice audio using speaker {args.speaker}: {args.output}")
except Exception as e:
    print(f"Error generating audio: {e}")
