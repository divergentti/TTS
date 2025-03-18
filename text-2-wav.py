"""
Divergentti 18.03.2025
Version 0.01

This is a NVIDIA CUDA implementation of the TTS.
NVIDIA NeMo:n FastPitch + HiFi-GAN vocoder
LJSpeech and Thorsten models.

Installation:
1. Make venv with command python -m venv venv
2. Activate it source venv/bin/activate
3. pip install Cython
4. pip install torch numpy soundfile
5. pip install nemo_toolkit

Note!
If you want to try the actual multi-speaker model, you might need to:

Create a Hugging Face account
Login with huggingface-cli login
Accept the model's terms of use on the Hugging Face website
"""


import torch
import numpy as np
import soundfile as sf
from torch.cuda import is_available as cuda_available
from pathlib import Path
import time

# Check for CUDA availability
if not cuda_available():
    print("CUDA is not available. Running on CPU instead.")
    device = "cpu"
else:
    device = "cuda"
    print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")


class NVidiaTTS:
    def __init__(self):
        try:
            # Import NVIDIA NeMo library for TTS
            from nemo.collections.tts.models import FastPitchModel, HifiGanModel

            # Models will automatically use GPU if available
            self.speaker_model = FastPitchModel.from_pretrained("nvidia/tts_en_fastpitch").to(device)
            self.vocoder = HifiGanModel.from_pretrained("nvidia/tts_hifigan").to(device)

            # Set models to evaluation mode
            self.speaker_model.eval()
            self.vocoder.eval()

            print("TTS models loaded successfully")

        except Exception as e:
            print(f"Error loading NeMo TTS models: {e}")
            raise RuntimeError("Failed to initialize TTS system")

    def generate_speech(self, text, output_path="output.wav", sample_rate=22050):
        """
        Convert text to speech and save to WAV file

        Args:
            text (str): The text to convert to speech
            output_path (str): Path to save the output audio file
            sample_rate (int): Sample rate of the output audio

        Returns:
            str: Path to the generated audio file
        """
        start_time = time.time()

        try:
            # Use NVIDIA NeMo for generation
            with torch.no_grad(), torch.cuda.amp.autocast(enabled=device == "cuda"):
                parsed = self.speaker_model.parse(text)
                spectrogram = self.speaker_model.generate_spectrogram(tokens=parsed)
                audio = self.vocoder.convert_spectrogram_to_audio(spec=spectrogram)

            # Convert from PyTorch tensor to numpy array
            audio_numpy = audio.to('cpu').numpy().squeeze()

            # Convert from float16 to float32 (soundfile doesn't support float16)
            if audio_numpy.dtype == np.float16:
                audio_numpy = audio_numpy.astype(np.float32)

            # Normalize audio to avoid clipping
            if np.max(np.abs(audio_numpy)) > 1.0:
                audio_numpy = audio_numpy / np.max(np.abs(audio_numpy))

            # Save the audio file
            output_path = Path(output_path)
            sf.write(output_path, audio_numpy, sample_rate)

        except Exception as e:
            print(f"Error in NeMo TTS generation: {e}")
            # Fallback to Coqui TTS
            try:
                self.tts.tts_to_file(text=text, file_path=output_path)
            except Exception as e:
                print(f"TTS generation failed: {e}")
                return None

        elapsed = time.time() - start_time
        print(f"Speech generated in {elapsed:.2f} seconds")
        return output_path

    def adjust_speech_parameters(self, speed=1.0, pitch_shift=0):
        """
        Adjust speech generation parameters

        Args:
            speed (float): Speed factor (0.5 to 2.0)
            pitch_shift (int): Pitch shift in semitones (-12 to 12)
        """
        try:
            self.speaker_model.set_speed(speed)
            self.speaker_model.set_pitch_shift(pitch_shift)
            print(f"Speech parameters adjusted: speed={speed}, pitch_shift={pitch_shift}")
        except:
            print("Parameter adjustment not available in fallback mode")


def main():
    """Example usage of the TTS system"""

    # Initialize the TTS system
    tts_engine = NVidiaTTS()

    # Example text
    text = "This voice synthesis is made with NVIDIA CUDA and with Sampo's old PC."

    # Generate speech
    output_file = tts_engine.generate_speech(text, "output_speech.wav")

    if output_file:
        print(f"Audio generated successfully: {output_file}")

        # Now try with different parameters
        tts_engine.adjust_speech_parameters(speed=0.9, pitch_shift=-2)
        tts_engine.generate_speech(text, "output_speech_modified.wav")


if __name__ == "__main__":
    main()
