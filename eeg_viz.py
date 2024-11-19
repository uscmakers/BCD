import serial
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import sounddevice as sd

class EEGVisualizer:
    def __init__(self, port='/dev/cu.usbmodem101', baudrate=9600, timeout=1):
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout

        # Initialize serial connection
        try:
            self.serial_conn = serial.Serial(self.port, self.baudrate, timeout=self.timeout)
        except serial.SerialException as e:
            raise Exception(f"Could not open serial port {self.port}: {e}")
        
        # EEG visualization parameters
        self.TIME_LENGTH = 1000
        self.raw_data_buffer = np.zeros(self.TIME_LENGTH)
        self.x_data = np.arange(self.TIME_LENGTH)
        self.data_count = 0
        
        # Add smoothing parameters
        self.WINDOW_SIZE = 5  # Adjust this value to change smoothing amount
        self.last_valid_value = 0
        self.max_change_threshold = 100  # Maximum allowed change between readings

        # Setup visualization
        plt.style.use('dark_background')
        self.fig, self.ax = plt.subplots(figsize=(12, 6))
        self.setup_plots()

    def setup_plots(self):
        self.ax.set_ylim(0, 2048)  # Arduino analog range
        self.ax.set_xlim(0, self.TIME_LENGTH)  # Set x-axis limits
        self.ax.set_title('EEG Signal')
        # Create line for the trailing signal
        self.line, = self.ax.plot([], [], 'g-', linewidth=1)
        # Create dot for the leading point
        self.point, = self.ax.plot([], [], 'go', markersize=10)
        # Add text display for current value
        self.value_display = self.ax.text(0.02, 0.95, '', transform=self.ax.transAxes, 
                                        color='white', fontsize=12)
        plt.tight_layout()

    def read_serial_data(self):
        """Read data from serial port and update buffer."""
        if self.serial_conn.in_waiting:
            try:
                data = int(self.serial_conn.readline().decode('utf-8').strip())
                
                # Ignore zero values
                if data == 0:
                    data = self.last_valid_value
                
                # Spike filtering
                if abs(data - self.last_valid_value) > self.max_change_threshold:
                    data = self.last_valid_value
                else:
                    self.last_valid_value = data

                if self.data_count < self.TIME_LENGTH:
                    # Fill from left to right until buffer is full
                    self.raw_data_buffer[self.data_count] = data
                    self.data_count += 1
                else:
                    # Once buffer is full, shift data left
                    self.raw_data_buffer[:-1] = self.raw_data_buffer[1:]
                    self.raw_data_buffer[-1] = data
                
                # Apply moving average smoothing
                if self.data_count >= self.WINDOW_SIZE:
                    end_idx = self.data_count if self.data_count < self.TIME_LENGTH else self.TIME_LENGTH
                    start_idx = max(0, end_idx - self.WINDOW_SIZE)
                    smoothed_value = np.mean(self.raw_data_buffer[start_idx:end_idx])
                    self.raw_data_buffer[end_idx-1] = smoothed_value

            except ValueError:
                pass

    def update(self, frame):
        self.read_serial_data()

        # Update trailing line (only show up to current data_count)
        if self.data_count < self.TIME_LENGTH:
            self.line.set_data(self.x_data[:self.data_count], 
                             self.raw_data_buffer[:self.data_count])
            self.point.set_data([self.x_data[self.data_count-1]], 
                              [self.raw_data_buffer[self.data_count-1]])
        else:
            self.line.set_data(self.x_data, self.raw_data_buffer)
            self.point.set_data([self.x_data[-1]], [self.raw_data_buffer[-1]])

        # Update value display
        current_value = self.raw_data_buffer[self.data_count-1 if self.data_count > 0 else 0]
        self.value_display.set_text(f'Value: {current_value:.0f}')

        return [self.line, self.point, self.value_display]

    def run(self):
        self.ani = FuncAnimation(self.fig, self.update, interval=20, blit=True)
        plt.show()

    def cleanup(self):
        if hasattr(self, 'serial_conn') and self.serial_conn.is_open:
            self.serial_conn.close()
        if hasattr(self, 'ani'):
            self.ani.event_source.stop()
        plt.close('all')

if __name__ == "__main__":
    try:
        visualizer = EEGVisualizer(port='/dev/cu.usbmodem101')  # Change 'COM3' to the correct port for your system
        visualizer.run()
    except Exception as e:
        print(f"Error: {e}")
    finally:
        visualizer.cleanup()
