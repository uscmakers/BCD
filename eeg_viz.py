import serial
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import sounddevice as sd
import scipy.signal as signal
from wavesynthstream import WaveSynth  # Import the WaveSynth class

class EEGVisualizer:
    def __init__(self, port='/dev/cu.usbmodem101', baudrate=115200):
        self.port = port
        self.baudrate = baudrate

        # Initialize serial connection
        try:
            self.serial_conn = serial.Serial(self.port, self.baudrate)
        except serial.SerialException as e:
            raise Exception(f"Could not open serial port {self.port}: {e}")
        
        # Adjust buffer for 250Hz sampling rate
        self.SAMPLE_RATE = 250  # Hz
        self.TIME_LENGTH = 250  # Reduced from 250 to 125 (0.5 seconds of data)
        self.raw_data_buffer = np.zeros(self.TIME_LENGTH)
        self.x_data = np.arange(self.TIME_LENGTH) / self.SAMPLE_RATE  # Time in seconds
        
        # Define frequency bands
        self.FREQ_BANDS = {
            'Delta': (0.5, 4),
            'Theta': (4, 8),
            'Alpha': (8, 13),
            'Beta': (13, 30),
            'Gamma': (30, 45)
        }
        
        # Setup plots
        self.fig, (self.ax_raw, self.ax_bands) = plt.subplots(2, 1, figsize=(15, 10))
        self.filtered_buffers = {band: np.zeros(self.TIME_LENGTH) for band in self.FREQ_BANDS}
        self.lines = {}
        self.setup_plots()

        # Initialize WaveSynth
        self.synth = WaveSynth()
        self.synth.start()

        # Pre-calculate filter coefficients
        self.filter_coeffs = {}
        for band_name, (low, high) in self.FREQ_BANDS.items():
            self.filter_coeffs[band_name] = signal.butter(
                4, [low, high], btype='bandpass', fs=self.SAMPLE_RATE
            )

    def setup_plots(self):
        # Set dark style for the entire figure
        self.fig.patch.set_facecolor('#1C1C1C')
        
        # Add padding to prevent cutoff
        plt.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.05)
        
        for ax in [self.ax_raw, self.ax_bands]:
            ax.set_facecolor('#2F2F2F')
            ax.grid(True, color='#444444')
            ax.tick_params(colors='white')
            ax.xaxis.label.set_color('white')
            ax.yaxis.label.set_color('white')
            for spine in ax.spines.values():
                spine.set_color('#444444')
        
        # Raw signal plot with adjusted limits
        self.ax_raw.set_ylim(0,300)
        self.ax_raw.set_xlim(0, self.TIME_LENGTH / self.SAMPLE_RATE)
        self.ax_raw.set_title('Raw EEG Signal', color='white', pad=20)
        self.line_raw, = self.ax_raw.plot([], [], color='#00FF00', linewidth=1, label='Raw')
        
        # Filtered signals plot with adjusted limits
        self.ax_bands.set_ylim(-50, 50)
        self.ax_bands.set_xlim(0, self.TIME_LENGTH / self.SAMPLE_RATE)
        self.ax_bands.set_title('Filtered Brain Waves', color='white', pad=20)
        
        colors = ['#00FFFF', '#00FF00', '#FF00FF', '#FFFF00', '#FF0000']
        for (band, _), color in zip(self.FREQ_BANDS.items(), colors):
            self.lines[band], = self.ax_bands.plot([], [], color=color, 
                                                 linewidth=1, label=band)
        
        self.ax_bands.legend(facecolor='#2F2F2F', labelcolor='white', 
                           edgecolor='#444444', loc='upper right')
        
        # Ensure proper layout
        self.fig.tight_layout()

    def read_serial_data(self):
        """Read data from serial port and update buffer."""
        if self.serial_conn.in_waiting:
            try:
                data = int(self.serial_conn.readline().decode('utf-8').strip())
                
                # Shift all data left by one position
                self.raw_data_buffer[:-1] = self.raw_data_buffer[1:]
                # Add new data point at the end
                self.raw_data_buffer[-1] = data

            except ValueError:
                pass

    def filter_data(self):
        """Optimized filter application."""
        for band_name, (b, a) in self.filter_coeffs.items():
            # Use lfilter instead of filtfilt for better performance
            self.filtered_buffers[band_name] = signal.lfilter(b, a, self.raw_data_buffer)

    def update(self, frame):
        self.read_serial_data()
        self.filter_data()

        # Batch update all line data at once
        self.line_raw.set_data(self.x_data, self.raw_data_buffer)
        
        # Use list comprehension for faster updates
        for band, line in self.lines.items():
            line.set_data(self.x_data, self.filtered_buffers[band])

        # Optimize frequency calculation
        self.synth.update_frequency(np.float64(self.raw_data_buffer[-1]))  # Use latest value instead of mean

        return [self.line_raw] + list(self.lines.values())

    def run(self):
        self.ani = FuncAnimation(self.fig, self.update, 
                               interval=1,  # Reduced from 2 to 1
                               blit=True,
                               cache_frame_data=False)
        plt.show()

    def cleanup(self):
        if hasattr(self, 'serial_conn') and self.serial_conn.is_open:
            self.serial_conn.close()
        if hasattr(self, 'ani'):
            self.ani.event_source.stop()
        plt.close('all')
        self.synth.cleanup()  # Ensure the synth is also cleaned up

if __name__ == "__main__":
    try:
        visualizer = EEGVisualizer(port='/dev/cu.usbserial-110')  # Change 'COM3' to the correct port for your system
        visualizer.run()
    except Exception as e:
        print(f"Error: {e}")
    finally:
        visualizer.cleanup()
 