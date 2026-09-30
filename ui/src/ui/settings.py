from typing import cast

import sounddevice as sd
from PyQt5 import QtCore, QtWidgets, uic

from session_manager import SessionManager
from theme import apply_theme, set_button_variant

class SettingsDialog(object):
    def __init__(self, session_manager: SessionManager):
        self._session_manager = session_manager
        self.devices_dict: dict[str, int] = {}

    def get_available_devices(self) -> list[str]:
        devices = sd.query_devices(kind="input")
        if isinstance(devices, list):
            return [str(device["name"]) for device in devices]
        return [str(devices["name"])]

    def create_device_dict(self) -> dict[str, int]:
        devices = sd.query_devices(kind="input")
        if isinstance(devices, list):
            return {str(device["name"]): int(device["index"]) for device in devices}
        return {str(devices["name"]): int(devices["index"])}

    def setupUi(self, dialog: QtWidgets.QDialog) -> None:
        uic.loadUi("ui/settings.ui", dialog)
        apply_theme(dialog)
        self.devices_dict = self.create_device_dict()
        self.timer = QtCore.QTimer(dialog)
        self.timer.timeout.connect(self.background_check)
        self.timer.start(3000)

        self.device_combo = cast(QtWidgets.QComboBox, dialog.findChild(QtWidgets.QComboBox, "device_combo"))
        self.model_combo = cast(QtWidgets.QComboBox, dialog.findChild(QtWidgets.QComboBox, "model_combo"))
        self.whisper_combo = cast(QtWidgets.QComboBox, dialog.findChild(QtWidgets.QComboBox, "whisper_combo"))
        self.gaze_combo = cast(QtWidgets.QComboBox, dialog.findChild(QtWidgets.QComboBox, "gaze_combo"))
        self.close_button = cast(QtWidgets.QPushButton, dialog.findChild(QtWidgets.QPushButton, "close_button"))

        self.device_combo.currentIndexChanged.connect(self.device_callback)
        self.whisper_combo.currentTextChanged.connect(self._on_whisper_mode_changed)
        self.gaze_combo.currentTextChanged.connect(self._on_gaze_mode_changed)
        self.close_button.clicked.connect(dialog.accept)
        set_button_variant(self.close_button, "secondary")

        self.device_combo.clear()
        self.device_combo.addItems(self.get_available_devices())
        self.model_combo.clear()
        self.model_combo.addItems(["small", "large"])

        self.whisper_combo.setCurrentText("Tryb online" if self._session_manager.whisper_online else "Tryb offline")
        self.gaze_combo.setCurrentText("Tryb online" if self._session_manager.gaze_online else "Tryb offline")
        self._on_whisper_mode_changed(self.whisper_combo.currentText())
        self._on_gaze_mode_changed(self.gaze_combo.currentText())

    def device_callback(self) -> None:
        selected = self.device_combo.currentText()
        if not selected:
            return
        selected_device = self.devices_dict[selected]
        self._session_manager.set_microphone_device(selected_device)
        print("Selected device index: ", selected_device)

    def _on_whisper_mode_changed(self, mode: str) -> None:
        self._session_manager.set_whisper_mode("online" in mode.lower())

    def _on_gaze_mode_changed(self, mode: str) -> None:
        self._session_manager.set_gaze_mode("online" in mode.lower())

    def background_check(self) -> None:
        devices = self.create_device_dict()
        if devices == self.devices_dict:
            return
        current_name = self.device_combo.currentText()
        self.devices_dict = devices
        self.device_combo.blockSignals(True)
        self.device_combo.clear()
        self.device_combo.addItems(self.get_available_devices())
        if current_name in self.devices_dict:
            self.device_combo.setCurrentText(current_name)
        self.device_combo.blockSignals(False)