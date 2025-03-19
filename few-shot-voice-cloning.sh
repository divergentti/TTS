# Few-shot voice cloning for Coqui-ai TTS
# Record your voice (Audacity etc) 5-30 minutes
# Speak naturally, make sure no bakcground noise
# Sepak at least:
# 	"The quick brown fox jumps over the lazy dog."
# 	"She sells seashells by the seashore."
#	"How now, brown cow?"
# More is better, read books from public domain books https://www.gutenberg.org/
# Export recorded sound to 44.1 kHz / 16-bit mono .wav
# Use the sound as speaker_wav
# In this example input is TTSJari.vaw
# Note: not all models support speaker_wav!

tts --text "$(cat text_to_be_spoken.txt)" \
    --model_name "tts_models/multilingual/multi-dataset/xtts_v2" \
    --use_cuda \
    --speaker_wav "TTSJari.wav" \
    --language_idx "en" \
    --out_path "jari_voice.wav"
