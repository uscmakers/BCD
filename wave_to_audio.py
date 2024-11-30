import numpy as np
import sounddevice as sd
import threading

# Parameters
duration = 5.0         # Duration in seconds
amplitude = 0.5       # Amplitude (0.0 to 1.0)
sample_rate = 44100   # Samples per second

# Frequency ranges for brainwaves
brainwave_frequencies = {
    "Delta": (0.3, 4),   # Delta waves (0.3-4 Hz)
    "Theta": (4, 8),     # Theta waves (4-8 Hz)
    "Alpha": (8, 13),    # Alpha waves (8-13 Hz)
    "Beta": (13, 30),    # Beta waves (13-30 Hz)
    "Gamma": (30, 100),  # Gamma waves (30 Hz and above)
}

# Function to generate a sine wave for each brainwave type
def generate_brainwave(frequency_range, duration, sample_rate, amplitude):
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    frequency = np.random.uniform(frequency_range[0], frequency_range[1])
    wave = amplitude * np.sin(2 * np.pi * frequency * t)
    return wave, frequency

# Function to play the audio of the brainwave
def play_audio(brainwave_data):
    sd.play(brainwave_data, samplerate=sample_rate)
    sd.wait()  # Wait for the brainwave audio to finish

# Function to generate and play the brainwave audio in parallel
def generate_and_play_brainwave(brainwave_name, frequency_range):
    brainwave_data, frequency = generate_brainwave(frequency_range, duration, sample_rate, amplitude)
    print(f"Generating {brainwave_name} wave with frequency {frequency/100:.2f} Hz")

    # Start the audio playback in a separate thread
    audio_thread = threading.Thread(target=play_audio, args=(brainwave_data,))
    audio_thread.start()

    # Wait for the audio to finish
    audio_thread.join()
    print(f"{brainwave_name} playback finished.")

# Run through each brainwave type
for brainwave_name, frequency_range in brainwave_frequencies.items():
    scaled_frequency_range = tuple(x * 100 for x in frequency_range)
    generate_and_play_brainwave(brainwave_name, scaled_frequency_range)
