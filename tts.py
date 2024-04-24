import numpy as np
from scipy.io.wavfile import write
import torch
from nemo.collections.tts.models.base import SpectrogramGenerator, Vocoder

if torch.cuda.is_available():
  device = "cuda"
else:
  device = "cpu"

# Load our spectrogram and vocoder models onto our CPU
spectrogram_model = SpectrogramGenerator.from_pretrained("tts_en_tacotron2").eval().to(device)
vocoder = Vocoder.from_pretrained("tts_en_hifigan").eval().to(device)


text = "Chainguard Images are built with security in mindfrom the ground up. They include fewer packages, a lighter footprint, included SBOMs, and active and ongoing CVE remediation."

# Normalize the text and convert it into individual phonemes/tokens.
tokens = spectrogram_model.parse(text, normalize=True)

# Generate spectrogram from text
spectrogram = spectrogram_model.generate_spectrogram(tokens=tokens)

# Invert the spectrogram into audio samples
audio = vocoder.convert_spectrogram_to_audio(spec=spectrogram)

# Convert output from pytorch tensor to numpy array
spectrogram = spectrogram.cpu().detach().numpy()[0]
audio = audio.cpu().detach().numpy()[0]

rate = 15000
scaled = np.int16(audio / np.max(np.abs(audio)) * 32767)
write('test.wav', rate, scaled)


