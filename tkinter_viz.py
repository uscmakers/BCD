import serial
import threading
import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import time

# Set the correct serial port and baud rate
PORT = "/dev/cu.usbserial-0001"  # Update this with your serial port
BAUDRATE = 9600

# Initialize serial connection
ser = serial.Serial(PORT, BAUDRATE, timeout=1)

# Initialize data lists for EEG frequency bands
time_values = []
delta_values = []
theta_values = []
low_alpha_values = []
high_alpha_values = []
low_beta_values = []
high_beta_values = []
low_gamma_values = []
high_gamma_values = []

# Initialize a time counter for the x-axis
time_counter = 0

# Function to read and parse serial data
def read_serial():
    global time_counter
    while True:
        try:
            if ser.in_waiting > 0:
                line = ser.readline().decode('utf-8').strip()  # Read and decode the serial data
                print(f"Received: {line}")
                
                # Split the data into a list
                data = line.split(",")
                print(f"Parsed data: {data}")  # Debugging: print parsed data
                
                # Check if the data contains enough values
                if len(data) >= 10:
                    # Extract and append the frequency band values
                    try:
                        delta_values.append(float(data[3]))  # Delta is the fourth value
                        theta_values.append(float(data[4]))  # Theta is the fifth value
                        low_alpha_values.append(float(data[5]))  # Low alpha is the sixth value
                        high_alpha_values.append(float(data[6]))  # High alpha is the seventh value
                        low_beta_values.append(float(data[7]))  # Low beta is the eighth value
                        high_beta_values.append(float(data[8]))  # High beta is the ninth value
                        low_gamma_values.append(float(data[9]))  # Low gamma is the tenth value
                        high_gamma_values.append(float(data[10]))  # High gamma is the eleventh value

                        # Debugging: print the extracted values for verification
                        print(f"Delta: {data[3]}, Theta: {data[4]}, Low Alpha: {data[5]}, High Alpha: {data[6]}, "
                              f"Low Beta: {data[7]}, High Beta: {data[8]}, Low Gamma: {data[9]}, High Gamma: {data[10]}")
                    except ValueError as e:
                        print(f"Error converting data to float: {e}")
                
                    # Append the current time index for the x-axis
                    time_values.append(time_counter)
                    
                    # Increment time counter
                    time_counter += 1
                else:
                    print("Received malformed data, skipping this entry.")
        except Exception as e:
            print(f"Error reading serial data: {e}")

# Function to plot the data in the tkinter window
def update_plot():
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot the values on the same graph with different lines
    ax.plot(time_values, delta_values, label="Delta", color="blue")
    ax.plot(time_values, theta_values, label="Theta", color="green")
    ax.plot(time_values, low_alpha_values, label="Low Alpha", color="red")
    ax.plot(time_values, high_alpha_values, label="High Alpha", color="orange")
    ax.plot(time_values, low_beta_values, label="Low Beta", color="purple")
    ax.plot(time_values, high_beta_values, label="High Beta", color="cyan")
    ax.plot(time_values, low_gamma_values, label="Low Gamma", color="magenta")
    ax.plot(time_values, high_gamma_values, label="High Gamma", color="brown")
    
    # Add titles and labels
    ax.set_title("EEG Frequency Bands Visualization")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Power")
    ax.legend()  # Show legend for each line
    
    # Embed the plot into the tkinter window
    canvas = FigureCanvasTkAgg(fig, master=frame)  # `frame` is your tkinter frame
    canvas.draw()
    canvas.get_tk_widget().pack()

    # Clear and update the plot every 1 second to keep it continuous
    root.after(1000, update_plot)

# Create a tkinter window
root = tk.Tk()
root.title("EEG Frequency Bands Visualization")

# Create a frame to hold the plot
frame = ttk.Frame(root)
frame.pack()

# Start the serial read in a separate thread
serial_thread = threading.Thread(target=read_serial)
serial_thread.daemon = True  # This allows the thread to close when the main program exits
serial_thread.start()

# Start updating the plot
update_plot()

# Run the tkinter main loop
root.mainloop()
