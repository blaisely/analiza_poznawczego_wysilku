import sys
from pathlib import Path
from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel, QWidget, 
                             QVBoxLayout, QHBoxLayout, QGridLayout, QPushButton, QStackedWidget)
from PyQt5 import uic
import os, random, string, json, time, hashlib

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiment import ExperimentWindow, MainWindow, EndWindow
from abc import abstractmethod, ABC
from devices.device import DeviceManager
from devices.microphone import Microphone
from eye_tracker.eye_tracker_recorder import EyeTrackerRecorder
from session_manager import SessionManager
from window_manager import WindowManager

def main():
    app = QApplication(sys.argv)
    QApplication.processEvents()
    setup_windows(app)
    sys.exit(app.exec_())

def setup_windows(app):
    eye_tracker = EyeTrackerRecorder()
    widget = QStackedWidget()
    device_manager = DeviceManager(eye_tracker)
    microphone  = Microphone()
    session_manager = SessionManager(device_manager=device_manager, micropohone=microphone)
    window = MainWindow()
    experiment_window = ExperimentWindow()
    end_window = EndWindow()
    window_manager = WindowManager(widget = widget, session_manager = session_manager, app=app)
    window_manager.add_window(window, "main")
    window_manager.add_window(experiment_window, "experiment")
    window_manager.add_window(end_window, "end")
    window_manager.start_window("main")
    
if __name__ == "__main__":
    main()
