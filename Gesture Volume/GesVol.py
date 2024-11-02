import warnings
import cv2
import time
import numpy as np
import HandTrackingModule as htm
import subprocess
import tkinter as tk
from tkinter import ttk, messagebox
from threading import Thread, Event, Lock
import platform
import logging
import os
from datetime import datetime


# Filter out the specific DeprecationWarning from protobuf
warnings.filterwarnings('ignore', category=UserWarning,
                       module='google.protobuf.symbol_database')


# Configure logging with more specific filters
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('volume_control.log'),
        logging.StreamHandler()
    ]
)

# Filter out specific warning messages from logging
logging.getLogger('mediapipe').setLevel(logging.ERROR)
logging.getLogger('google.protobuf').setLevel(logging.ERROR)


# [Previous imports remain the same...]

class VolumeController:
    """Handles volume control operations with specific support for Fedora."""

    def __init__(self):
        self.os_type = platform.system().lower()
        self.volume_lock = Lock()

    def set_volume(self, volume_percentage):
        """Set system volume using amixer command optimized for Fedora."""
        with self.volume_lock:
            volume_percentage = max(0, min(volume_percentage, 100))
            try:
                if self.os_type == 'linux':
                    # Simplified amixer command that works on Fedora
                    subprocess.run(
                        ['amixer', 'set', 'Master', f'{int(volume_percentage)}%'],
                        check=True,
                        capture_output=True,
                        text=True
                    )
                    return True
                elif self.os_type == 'windows':
                    # [Windows code remains the same...]
                    pass
                elif self.os_type == 'darwin':
                    # [macOS code remains the same...]
                    pass
                return False
            except Exception as e:
                logging.error(f"Error setting volume: {e}")
                # Try alternative command if the first one fails
                try:
                    subprocess.run(
                        ['amixer', '-M', 'set', 'Master', f'{int(volume_percentage)}%'],
                        check=True,
                        capture_output=True,
                        text=True
                    )
                    return True
                except Exception as e2:
                    logging.error(f"Error setting volume with alternative command: {e2}")
                    return False
    def get_current_volume(self):
        """Get current system volume."""
        try:
            result = subprocess.run(
                ['amixer', 'get', 'Master'],
                check=True,
                capture_output=True,
                text=True
            )
            # Parse the output to get current volume
            for line in result.stdout.splitlines():
                if 'Playback' in line and '%' in line:
                    volume = int(line.split('[')[1].split('%')[0])
                    return volume
        except Exception as e:
            logging.error(f"Error getting current volume: {e}")
            return 50  # Return default value if unable to get current volume
        return 50
class VolumeControlApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Gesture Volume Control")
        self.root.geometry("300x300")
        self.setup_gui()

        # Initialize controllers and state
        self.volume_controller = VolumeController()
        self.detector = htm.HandDetector(detection_confidence=0.7, max_hands=1)
        self.running = False
        self.volume_percentage = 0
        self.is_adjusting = False
        self.stop_event = Event()
        self.camera_index = 0
        self.sensitivity = 1.0

        # Initialize with current system
        self.volume_percentage = self.volume_controller.get_current_volume()
        self.update_volume_display(self.volume_percentage)

        # Performance monitoring
        self.fps_history = []
        self.last_volume_change = time.time()
        self.volume_change_cooldown = 0.1  # seconds

        # Settings
        self.settings = {
            'min_distance': 50,
            'max_distance': 200,
            'smoothing': 0.5,
            'gesture_timeout': 2.0
        }

    def setup_gui(self):
        """Setup the enhanced GUI with multiple tabs and controls."""
        # Create notebook for tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(expand=True, fill='both', padx=5, pady=5)

        # Main control tab
        self.main_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.main_frame, text='Control')

        # Volume display
        self.volume_label = ttk.Label(self.main_frame, text="Volume: 0%", font=("Helvetica", 20))
        self.volume_label.pack(pady=20)

        # Custom volume bar with gradient
        self.volume_bar = tk.Canvas(self.main_frame, width=400, height=40, bg='#f0f0f0')
        self.volume_bar.pack(pady=10)

        # Control buttons
        btn_frame = ttk.Frame(self.main_frame)
        btn_frame.pack(pady=20)

        self.start_button = ttk.Button(btn_frame, text="Start Tracking", command=self.start_tracking)
        self.start_button.pack(side=tk.LEFT, padx=5)

        self.stop_button = ttk.Button(btn_frame, text="Stop Tracking", command=self.stop_tracking, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=5)

        # Status indicator
        self.status_label = ttk.Label(self.main_frame, text="Status: Stopped", font=("Helvetica", 12))
        self.status_label.pack(pady=10)

        # Settings tab
        self.settings_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.settings_frame, text='Settings')

        # Camera selection
        ttk.Label(self.settings_frame, text="Camera:").pack(pady=5)
        self.camera_var = tk.StringVar(value="0")
        self.camera_combo = ttk.Combobox(self.settings_frame, textvariable=self.camera_var, values=self.get_available_cameras())
        self.camera_combo.pack(pady=5)

        # Sensitivity control
        ttk.Label(self.settings_frame, text="Sensitivity:").pack(pady=5)
        self.sensitivity_scale = ttk.Scale(self.settings_frame, from_=0.1, to=2.0, orient=tk.HORIZONTAL)
        self.sensitivity_scale.set(1.0)
        self.sensitivity_scale.pack(pady=5)

        # Debug information
        self.debug_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.debug_frame, text='Debug')

        self.debug_text = tk.Text(self.debug_frame, height=10, width=50)
        self.debug_text.pack(pady=10, padx=10)

        # Style configuration
        self.style = ttk.Style()
        self.style.configure('TButton', padding=10)

    def get_available_cameras(self):
        """Detect available cameras."""
        available_cameras = []
        for i in range(5):  # Check first 5 indexes
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                available_cameras.append(str(i))
                cap.release()
        return available_cameras if available_cameras else ["0"]

    def update_volume_display(self, volume_percentage):
        """Update volume display with smooth gradient effect."""
        self.volume_label.config(text=f"Volume: {int(volume_percentage)}%")
        self.volume_bar.delete("all")

        # Create gradient effect
        width = 400
        height = 40
        segments = 100
        for i in range(segments):
            x1 = i * (width / segments)
            x2 = (i + 1) * (width / segments)
            if x1 <= (volume_percentage / 100 * width):
                # Gradient from blue to lighter blue
                intensity = int(155 + (i / segments * 100))
                color = f'#{0:02x}{intensity:02x}ff'
                self.volume_bar.create_rectangle(x1, 0, x2, height, fill=color, outline="")

    def start_tracking(self):
        """Start the hand tracking process."""
        if not self.running:
            try:
                self.running = True
                self.stop_event.clear()
                self.camera_index = int(self.camera_var.get())
                self.sensitivity = self.sensitivity_scale.get()

                Thread(target=self.track_hand, daemon=True).start()

                self.start_button.configure(state=tk.DISABLED)
                self.stop_button.configure(state=tk.NORMAL)
                self.status_label.config(text="Status: Running")
                logging.info("Hand tracking started")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to start tracking: {str(e)}")
                logging.error(f"Failed to start tracking: {e}")

    def stop_tracking(self):
        """Stop the hand tracking process."""
        self.running = False
        self.stop_event.set()
        self.start_button.configure(state=tk.NORMAL)
        self.stop_button.configure(state=tk.DISABLED)
        self.status_label.config(text="Status: Stopped")
        logging.info("Hand tracking stopped")

    def track_hand(self):
        """Main hand tracking loop with improved error handling and performance monitoring."""
        cap = None
        try:
            cap = cv2.VideoCapture(self.camera_index)
            if not cap.isOpened():
                raise RuntimeError("Unable to access the camera")

            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

            previous_time = time.time()
            smoothed_volume = self.volume_percentage

            while self.running and not self.stop_event.is_set():
                success, img = cap.read()
                if not success:
                    logging.warning("Failed to capture image")
                    continue

                # Process frame
                img = self.detector.find_hands(img)
                lm_list, bbox = self.detector.find_position(img, draw=False)

                if lm_list:
                    # Calculate hand area for depth estimation
                    area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) // 100

                    # Get finger positions
                    fingers = self.detector.fingers_up()

                    # Check for volume control gesture
                    if fingers[0] and fingers[1] and not any(fingers[2:]):
                        if not self.is_adjusting:
                            self.is_adjusting = True
                            logging.info("Volume adjustment started")

                        # Get thumb-index distance
                        length, img, line_info = self.detector.find_distance(4, 8, img)
                        if length is not None:
                            # Apply sensitivity and smoothing
                            target_volume = np.interp(length * self.sensitivity,
                                                    [self.settings['min_distance'],
                                                     self.settings['max_distance']],
                                                    [0, 100])

                            # Smooth the volume changes
                            smoothed_volume = (smoothed_volume * self.settings['smoothing'] +
                                            target_volume * (1 - self.settings['smoothing']))

                            # Apply volume change cooldown
                            current_time = time.time()
                            if current_time - self.last_volume_change >= self.volume_change_cooldown:
                                self.volume_percentage = int(smoothed_volume)
                                if self.volume_controller.set_volume(self.volume_percentage):
                                    self.root.after(0, self.update_volume_display, self.volume_percentage)
                                    self.last_volume_change = current_time

                    # Reset adjustment flag when gesture ends
                    elif self.is_adjusting:
                        self.is_adjusting = False
                        logging.info("Volume adjustment ended")

                # Calculate and log FPS
                current_time = time.time()
                fps = 1 / (current_time - previous_time)
                previous_time = current_time
                self.fps_history.append(fps)

                # Update debug information periodically
                if len(self.fps_history) >= 30:
                    avg_fps = sum(self.fps_history) / len(self.fps_history)
                    self.update_debug_info(f"Average FPS: {avg_fps:.1f}")
                    self.fps_history = []

        except Exception as e:
            logging.error(f"Error in hand tracking: {e}")
            self.root.after(0, messagebox.showerror, "Error", f"Tracking error: {str(e)}")
        finally:
            if cap is not None:
                cap.release()
            self.root.after(0, self.stop_tracking)

    def update_debug_info(self, info):
        """Update debug information display."""
        self.debug_text.insert(tk.END, f"{datetime.now().strftime('%H:%M:%S')}: {info}\n")
        self.debug_text.see(tk.END)
        if self.debug_text.index('end-1c').split('.')[0] > '100':
            self.debug_text.delete('1.0', '50.0')

    def on_closing(self):
        """Clean up resources before closing."""
        self.stop_tracking()
        logging.info("Application closing")
        self.root.destroy()

if __name__ == "__main__":
    try:
        root = tk.Tk()
        app = VolumeControlApp(root)
        root.protocol("WM_DELETE_WINDOW", app.on_closing)
        root.mainloop()
    except Exception as e:
        logging.critical(f"Application failed to start: {e}")
        raise
