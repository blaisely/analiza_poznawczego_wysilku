import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel, QWidget, 
                             QVBoxLayout, QHBoxLayout, QGridLayout, QPushButton, QStackedWidget)
from PyQt5 import uic
import os, random, string, json, time, hashlib
from PyQt5.QtCore import QThread
from eye_tracker.eye_tracker_recorder import EyeTrackerRecorder

class DeviceManager:
    def __init__(self, device: EyeTrackerRecorder = None):
        self.device = device

    def set_path(self, folder_path):
        self.device.set_path(folder_path)

    def run(self):
        self.device.start()

    def stop(self, *, online: bool = True):
        self.device.process_session(session_suffix = "tokens", online=online)
