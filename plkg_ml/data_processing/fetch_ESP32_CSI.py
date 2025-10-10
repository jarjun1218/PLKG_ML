from time import sleep
import serial
import threading
import numpy as np
from datetime import datetime
import math
from scipy.signal import savgol_filter
import platform
import os

# matplotlib for plotting
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# CSIReader for MacOS
class CSIReader:
    def __init__(self, port='', baud=115200, timeout=0.1):
        # Initialize the serial port
        self.ser = serial.Serial(port, baud, timeout=timeout)

        # Clear the input and output buffers
        self.ser.reset_input_buffer()
        self.ser.reset_output_buffer()

        # Set a lock for reading the serial data
        self._thread = threading.Thread(target=self._monitor, daemon=True)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        
        # Initialize variables
        self.no_used_carriers_ch1 = [0, 1, 2, 3, 4, 5, 11, 32, 59, 60, 61, 62, 63]
        self.no_used_carriers_ch13 = [0, 1, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37]
        self.eve_no_used_carriers = [0, 1, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37]
        self.serial_num = 0
        self.timestamp = datetime.now().strftime("%H:%M:%S") + '.%02d' % (datetime.now().microsecond // 10000)
        self.rssi = 0
        self.noise = 0
        self.mac = ""
        self.device = ""
        self.csi_amp_array = np.zeros(51, dtype=np.float32)
        self.csi_amp_filtered = np.zeros(51, dtype=np.float32)
        self.combined = np.zeros(54, dtype=np.float32)

    def start(self):
        self._stop.clear()
        if not self._thread.is_alive():
            self._thread.start()
            
    def stop(self):
        self._stop.set()
        self._thread.join(timeout=1)
        if self.ser.is_open:
            self.ser.reset_input_buffer()
            self.ser.reset_output_buffer()
            self.ser.close()
            print("Serial port closed.")
        
    def _monitor(self):
        try:
            while not self._stop.is_set():
                if not self.ser.is_open:
                    break
                # Read a line from the serial port
                line = self.ser.readline().decode(errors='ignore').strip()  # Read a line from the serial port
                self.timestamp = datetime.now().strftime("%H:%M:%S") + '.%02d' % (datetime.now().microsecond // 10000)  # Get current timestamp
                if "serial_num:" in line:
                    data = line.split(',')[1:]  # Extract CSI data from the line
                    self.serial_num = int(data[0])  # Serial number
                    self.rssi = int(data[2])  # RSSI value
                    self.noise = int(data[3])  # Noise value
                    if data[1] == '384' or data[1] == '256': #GCS
                        self.device = "GCS"
                        csi_amp = np.array(data[5:-1], dtype=np.float32)  # CSI amplitude values
                        if len(csi_amp) == 64:
                            csi_amp = np.delete(csi_amp, self.no_used_carriers_ch1)
                            # print(f"{self.device} {self.serial_num} Raw CSI Data: {csi_amp} len: {len(csi_amp)}\n")
                        elif len(csi_amp) == 128:
                            # this is not calculated yet so it should be calculated
                            for i in range(64):
                                csi_amp[i] = math.sqrt(csi_amp[2*i]**2 + csi_amp[2*i+1]**2)
                            csi_amp = csi_amp[:64]
                            csi_amp = np.delete(csi_amp, self.no_used_carriers_ch13)
                        else:
                            print(f"Invalid {self.device}'s CSI data length: {len(csi_amp)}")
                            continue
                    elif data[1] == '128': #UAV
                        self.device = "UAV"
                        csi_amp = np.array(data[5:-1], dtype=np.float32)  # CSI amplitude values
                        if len(csi_amp) == 64:
                            csi_amp = np.delete(csi_amp, self.no_used_carriers_ch1)
                            # print(f"{self.device} {self.serial_num} Raw CSI Data: {csi_amp} len: {len(csi_amp)}\n")
                        elif len(csi_amp) == 128:
                            # this is not calculated yet so it should be calculated
                            for i in range(64):
                                csi_amp[i] = math.sqrt(csi_amp[2*i]**2 + csi_amp[2*i+1]**2)
                            csi_amp = csi_amp[:64]
                            csi_amp = np.delete(csi_amp, self.no_used_carriers_ch13)
                        else:
                            print(f"Invalid {self.device}'s CSI data length: {len(csi_amp)}")
                            continue
                    
                    # Normalize the CSI data
                    csi_amp = (csi_amp - 0) / (61.846 - 0) # Normalize to 0-100 range
                    # csi_amp = (csi_amp - 0) / (30 - 0) * 100 # Normalize to 0-100 range
                    
                    # Writing to the array in a thread-locked manner
                    with self._lock:
                        if len(csi_amp) == 51:
                            self.csi_amp_array = csi_amp

                            # Apply Savitzky-Golay filter to smooth the data
                            # self.csi_amp_filtered = savgol_filter(csi_amp, window_length=10, polyorder=2)
                            # self.csi_amp_filtered = (self.csi_amp_filtered - 0) / (61.846 - 0) # Normalize
                            # self.csi_amp_filtered = self.csi_amp_filtered.astype(np.float32) # Convert to float32
                            self.combined = np.concatenate(([self.serial_num, self.rssi, self.noise], csi_amp))
                        else:
                            print(f"Invalid {self.device}'s CSI data length: {len(csi_amp)}")
                elif "eve," in line:
                    data = line.split(',')[1:]  # Extract CSI data from the line
                    self.rssi = int(data[2])  # RSSI value
                    self.noise = int(data[3])  # Noise value
                    self.device = "EVE"
                    self.mac = data[4]  # TX MAC address
                    csi_amp = np.array(data[5:], dtype=np.float32)  # CSI amplitude values
                    csi_amp = np.delete(csi_amp, self.eve_no_used_carriers)
                    # print(f"Eve's Raw CSI Data: {csi_amp} len: {len(csi_amp)}\n")

                    # Normalize the CSI data
                    csi_amp = (csi_amp - 0) / (61.846 - 0) # Normalize to 0-100 range
                    # csi_amp = (csi_amp - 0) / (30 - 0) * 100 # Normalize to 0-100 range

                    with self._lock:
                        if len(csi_amp) == 51:
                            self.csi_amp_array = csi_amp

                            # Apply Savitzky-Golay filter to smooth the data
                            # self.csi_amp_filtered = savgol_filter(csi_amp, window_length=10, polyorder=2)
                            # self.csi_amp_filtered = (self.csi_amp_filtered - 0) / (61.846 - 0) # Normalize
                            # self.csi_amp_filtered = self.csi_amp_filtered.astype(np.float32) # Convert to float32
                            self.combined = np.concatenate(([0, self.rssi, self.noise], csi_amp))
                        else:
                            print(f"Invalid Eve's CSI data length: {len(csi_amp)}")
                else:
                    pass
        except serial.SerialException as e:
            print(f"{self.device} Serial error: {e}")
        except Exception as e:
            print(f"{self.device} Error: {e}")
            pass
        finally:
            # Ensure the serial port is closed properly
            if self.ser.is_open:
                self.ser.close()
                print(f"{self.device} Serial port closed.")
            else:
                print(f"{self.device} Serial port already closed.")
    
    # real-time plot using matplotlib animation
    def start_plot(self, interval=100):
        self.fig, self.ax = plt.subplots()
        x = np.arange(self.csi_amp_array.size)
        self.line, = self.ax.plot(x, self.csi_amp_array)
        self.ax.set_xlim(0, self.csi_amp_array.size - 1)
        self.ax.set_ylim(0, 100)
        plt.xlabel('Subcarrier Index')
        plt.ylabel('Amplitude')
        plt.title('CSI Amplitude')
        plt.grid()
        
        self.ani = animation.FuncAnimation(
            self.fig, 
            self.update_plot, 
            interval=interval,
            cache_frame_data=False,
        )
        plt.tight_layout()
        plt.show()
        
    def update_plot(self, frame):
        with self._lock:
            data = self.csi_amp_array.copy()
        if data.size != self.line.get_ydata().size:
            # print("Data size mismatch, skipping update.")
            return (self.line,)
        else:
            self.line.set_ydata(data)
            self.ax.set_ylim(0.1, max(0.5, data.max() * 1.2))  # Adjust y-limits dynamically
            return (self.line,)

    def stop_plot(self):
        if self.ani:
            plt.close(self.fig)
            print("Plot closed.")
        else:
            print("No plot to close.")
                
if __name__ == "__main__":
    if platform.system() == "Darwin":  # MacOS
        reader = CSIReader(port='/dev/cu.usbserial-3', baud=115200)
    elif platform.system() == "Windows":
        reader = CSIReader(port='COM4', baud=115200)
    elif platform.system() == "Linux":
        reader = CSIReader(port='/dev/ttyUSB0', baud=115200)
    else:
        raise EnvironmentError("Unsupported platform")
    reader.start()
    try:
        reader.start_plot(interval=100)  # Start the plot with a 100ms interval
    except KeyboardInterrupt:
        reader.stop()
    finally:
        reader.stop()
        print("Program terminated.")
        reader.stop_plot()
        
        