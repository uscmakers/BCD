from pyo import *
import numpy as np
import time

# Initialize the server
s = Server().boot()
s.start()

# Frequency ranges for brainwaves
brainwave_frequencies = {
    "Delta": (0.3, 4),   # Delta waves (0.3-4 Hz)
    "Theta": (4, 8),     # Theta waves (4-8 Hz)
    "Alpha": (8, 13),    # Alpha waves (8-13 Hz)
    "Beta": (13, 30),    # Beta waves (13-30 Hz)
    "Gamma": (30, 100),  # Gamma waves (30-100 Hz)
}

# Parameters
duration = 5.0  # Duration for each brainwave in seconds
amplitude = 0.5  # Synth volume
base_note = 440  # A4 note in Hz
scale_intervals = [0, 2, 4, 5, 7, 9, 11]  # Major scale intervals
num_notes = 8  # Number of notes in our random sequence

# Generate synth sounds for brainwaves
def generate_synth(brainwave_name, frequency_range):
    mod_frequency = np.random.uniform(*frequency_range)
    print(f"{brainwave_name}: Modulation Frequency {mod_frequency:.2f} Hz")

    # Generate random melody sequence within two octaves
    melody_notes = []
    for _ in range(num_notes):
        # Choose random scale degree and octave
        scale_degree = np.random.choice(scale_intervals)
        octave = np.random.randint(0, 2)  # 0 or 1 for two octaves
        semitones = scale_degree + (12 * octave)
        melody_notes.append(semitones)
    
    print(f"Melody sequence (semitones): {melody_notes}")

    # Use logarithmic scaling for better proportional mapping
    log_freq = np.log10(mod_frequency)
    log_min = np.log10(frequency_range[0])
    log_max = np.log10(frequency_range[1])
    
    # Normalize to 0-1 range using log scale
    tempo = 1.0 - (log_freq - log_min) / (log_max - log_min)
    metro_time = float(tempo * 0.9 + 0.1)  # Scale to range 0.1 to 1.0 seconds
    print(f"Tempo interval: {metro_time:.3f} seconds")

    # Create drum sounds with improved settings
    # Hi-hat: brighter and punchier
    hihat = Noise(mul=0.15)
    hihat_filter = ButHP(hihat, freq=9000)  # Higher frequency for more sizzle
    hihat_env = Adsr(attack=0.001, decay=0.03, sustain=0, release=0.01)  # Shorter decay
    hihat_out = hihat_filter * hihat_env
    
    # Kick: deeper and punchier
    kick = Sine(freq=50, mul=0.5)  # Lower frequency, higher amplitude
    kick_env = Adsr(attack=0.001, decay=0.15, sustain=0, release=0.01)
    kick_pitch_env = Adsr(attack=0.001, decay=0.05, sustain=0, release=0.01)
    kick_pitch = Port(kick_pitch_env, risetime=0.001, falltime=0.001, mul=100, add=50)  # More pitch movement
    kick.freq = kick_pitch
    kick_out = kick * kick_env
    
    # More natural drum patterns
    drum_metro = Metro(time=metro_time).play()
    # Simple but effective hi-hat pattern (emphasis on offbeats)
    hihat_pat = Beat(time=drum_metro, taps=8, w1=[60,90,60,90,60,90,60,90])
    # Basic kick pattern (four-on-the-floor with variations)
    kick_pat = Beat(time=drum_metro, taps=8, w1=[100,0,60,0,90,0,60,30])
    
    # Trigger drum envelopes
    hihat_trig = TrigFunc(hihat_pat, hihat_env.play)
    kick_trig = TrigFunc(kick_pat, function=[kick_env.play, kick_pitch_env.play])
    
    # Mix drums with better balance
    drums_mix = Mix([hihat_out, kick_out], voices=2, mul=0.7).out()  # Increased overall drum volume

    # Reduce melody volume slightly to make drums more prominent
    melody = Sine(freq=base_note, mul=amplitude * 0.4).out()
    
    # Create a table of frequencies for our random melody
    freqs = []
    for semitone in melody_notes:
        freq = base_note * pow(2, semitone/12.0)  # Convert semitones to frequency
        freqs.append(freq)
    
    freq_table = DataTable(size=len(freqs), init=freqs)
    count = Counter(Metro(time=metro_time).play(), min=0, max=len(freqs)-1)
    freq = TableIndex(freq_table, count)
    
    # Combine the base frequency with modulation
    final_freq = SigTo(freq) + modulator
    melody.freq = final_freq

    # Play the sound for the given duration
    time.sleep(duration)
    
    # Stop all sounds
    melody.stop()
    count.stop()
    drums_mix.stop()
    drum_metro.stop()
    hihat_pat.stop()
    kick_pat.stop()

# Main logic
def main():
    scaled_ranges = {k: (v[0] * 100, v[1] * 100) for k, v in brainwave_frequencies.items()}  # Scale to Hz
    for brainwave_name, frequency_range in scaled_ranges.items():
        generate_synth(brainwave_name, frequency_range)

if __name__ == "__main__":
    try:
        main()
    finally:
        s.stop()
        s.shutdown()
