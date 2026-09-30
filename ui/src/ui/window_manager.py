import sys
from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel, QWidget, 
                             QVBoxLayout, QHBoxLayout, QGridLayout, QPushButton, QStackedWidget)
from ui.session_manager import SessionManager


class WindowManager:
    def __init__(self, widget: QMainWindow, session_manager: SessionManager, app):
        self.session_manager = session_manager
        self.widget = widget
        self.current_window = 0
        self.windows = {}
        self.app = app
        # self.widget.setMinimumSize(800,500)
        self.widget.setFixedSize(1920, 1200)

    def next_window(self):
        self.current_window += 1
        if self.current_window >= self.widget.count():
            print("This is the last window!")
            self.current_window = self.widget.count() - 1
            return
        self.widget.setCurrentIndex(self.current_window)
        current = self.widget.currentWidget()
        
        if hasattr(current, "init"):
            current.init()

    def previous_window(self):
        if self.current_window() < 0:
            self.current_window -= 1
            self.widget.setCurrentIndex(self.current_window)
            if hasattr(self.widget.currentWidget(), "init"):
                self.widget.currentWidget().init()
        else:
            print("This is the last window!")
            return
    def get_windows(self):
        return self.windows.keys()
    
    def change_window(self, name : str):
        if name in self.windows:
            self.widget.setCurrentWidget(self.windows[name])
            print(self.windows[name])
            self.windows[name].init()
            print(self.widget.currentWidget())

    def add_window(self, window : QMainWindow, name: str):
        self.windows[name] = window
        self.widget.addWidget(window)
        window.add_session_manager(self.session_manager)
        window.add_window_manager(self)
        window.add_app(self.app)
        
    def start_window(self, name : str):
        if self.windows is not None:
            self.widget.setCurrentWidget(self.windows[name])
        self.widget.show()