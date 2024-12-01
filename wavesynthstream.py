from pyo import *
import numpy as np
import time
from queue import Queue
from threading import Thread, Event

class WaveSynth:
    def __init__(self):
        # Initialize the server
        self.server = Server().boot()
        self.server.start()
        
        # Sound parameters
        self.base_note = 440
        self.amplitude = 0.5
        self.scale_intervals = [0, 2, 4, 5, 7, 9, 11]
        self.num_notes = 8
        
        # Control flags
        self.is_playing = Event()
        self.current_frequency = 0
        
        # Initialize sound components
        self._setup_sounds()

    def _setup_sounds(self):
        # Generate random melody sequence
        self.melody_notes = []
        for _ in range(self.num_notes):
            scale_degree = np.random.choice(self.scale_intervals)
            octave = np.random.randint(0, 2)
            semitones = scale_degree + (12 * octave)
            self.melody_notes.append(semitones)

        # Create the melody oscillator
        self.melody = Sine(freq=self.base_note, mul=self.amplitude * 0.4).out()
        self.modulator = Sine(freq=1, mul=20, add=0)

        # Setup drums
        # Hi-hat
        self.hihat = Noise(mul=0.15)
        self.hihat_filter = ButHP(self.hihat, freq=9000)
        self.hihat_env = Adsr(attack=0.001, decay=0.03, sustain=0, release=0.01)
        self.hihat_out = self.hihat_filter * self.hihat_env
        
        # Kick
        self.kick = Sine(freq=50, mul=0.5)
        self.kick_env = Adsr(attack=0.001, decay=0.15, sustain=0, release=0.01)
        self.kick_pitch_env = Adsr(attack=0.001, decay=0.05, sustain=0, release=0.01)
        self.kick_pitch = Port(self.kick_pitch_env, risetime=0.001, falltime=0.001, mul=100, add=50)
        self.kick.freq = self.kick_pitch
        self.kick_out = self.kick * self.kick_env
        
        # Initialize metro for timing
        self.metro = Metro(time=0.5).play()
        
        # Setup patterns
        self.hihat_pat = Beat(time=self.metro, taps=8, w1=[60,90,60,90,60,90,60,90])
        self.kick_pat = Beat(time=self.metro, taps=8, w1=[100,0,60,0,90,0,60,30])
        
        # Trigger drum envelopes
        self.hihat_trig = TrigFunc(self.hihat_pat, self.hihat_env.play)
        self.kick_trig = TrigFunc(self.kick_pat, function=[self.kick_env.play, self.kick_pitch_env.play])
        
        # Mix drums
        self.drums_mix = Mix([self.hihat_out, self.kick_out], voices=2, mul=0.7).out()

        # Setup melody sequencer
        freqs = [self.base_note * pow(2, st/12.0) for st in self.melody_notes]
        self.freq_table = DataTable(size=len(freqs), init=freqs)
        self.count = Counter(self.metro, min=0, max=len(freqs)-1)
        self.freq = TableIndex(self.freq_table, self.count)
        
        # Combine frequency sources
        self.final_freq = SigTo(self.freq) + self.modulator
        self.melody.freq = self.final_freq

    def update_frequency(self, new_freq):
        """Update the synth based on new frequency input"""
        if not self.is_playing.is_set():
            return

        # Update modulator frequency
        self.modulator.freq = new_freq
        
        # Calculate new tempo (logarithmic scaling)
        log_freq = np.log10(max(new_freq, 0.1))  # Prevent log(0)
        log_min = np.log10(0.1)   # Minimum expected frequency
        log_max = np.log10(100.0) # Maximum expected frequency
        
        # Scale to reasonable tempo range
        tempo = 1.0 - (log_freq - log_min) / (log_max - log_min)
        metro_time = float(tempo * 0.9 + 0.1)  # Scale to range 0.1 to 1.0 seconds
        
        # Update metro time
        self.metro.time = metro_time

    def start(self):
        """Start the synth"""
        self.is_playing.set()

    def stop(self):
        """Stop the synth"""
        self.is_playing.clear()
        self.melody.stop()
        self.drums_mix.stop()
        self.metro.stop()
        self.hihat_pat.stop()
        self.kick_pat.stop()

    def cleanup(self):
        """Clean up and shut down"""
        self.stop()
        self.server.stop()
        self.server.shutdown()

# Example usage:
if __name__ == "__main__":
    synth = WaveSynth()
    synth.start()
    
    try:
        # Example: simulate incoming frequency data
        while True:
            # Simulate new frequency data (replace this with your actual frequency input)
            test_freq = np.random.uniform(0.1, 100.0)
            synth.update_frequency(test_freq)
            time.sleep(0.1)  # Update rate
            
    except KeyboardInterrupt:
        print("\nStopping synth...")
    finally:
        synth.cleanup()