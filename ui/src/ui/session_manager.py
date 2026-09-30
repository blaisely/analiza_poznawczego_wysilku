from typing import Any, List, Optional

from utils.utils import Utils
import os
import json
import time
from devices.device import DeviceManager
from devices.microphone import Microphone

class SessionManager:
    def __init__(self, device_manager: DeviceManager, micropohone: Microphone, base_dir: str = "data"):
        self.base_dir = base_dir
        self.participant_id = None
        self.participant_data = None
        self.session_path: Optional[str] = None
        self.current_path: Optional[str] = None
        self.tokenizer_name: Optional[str] = None
        self._device_manager = device_manager
        self.n_recording = 0
        self.microphone = micropohone
        self._whisper_online = True
        self._gaze_online = True

    def add_device_manager(self, device_manager: DeviceManager):
        self._device_manager = device_manager

    def start_new_session(self, participant_data: dict) -> None:
        if not os.path.exists("./data"):
            os.makedirs("./data")
        self.participant_data = participant_data
        self.session_path, self.participant_id = Utils.create_participant_data(participant_data)

    def create_question_folder(self) -> Optional[str]:
        if not self.session_path:
            print("Cannot create question folder: session path not initialized.")
            return None
        self.n_recording += 1
        self.current_path = os.path.join(self.session_path, f"Q_{self.n_recording}")
        os.makedirs(self.current_path, exist_ok=True)
        return self.current_path

    def start_capturing(self) -> None:
        if not self.current_path:
            folder = self.create_question_folder()
            if not folder:
                return
        self._set_device_path()
        self._device_manager.run()

    def set_microphone_device(self, index: int) -> None:
        self.microphone.set_device(index)

    def _set_device_path(self) -> None:
        if not self.current_path:
            print("Device path not set because current_path is missing.")
            return
        print("Current Session path: \n", self.current_path)
        self._device_manager.set_path(self.current_path)

    def stop_capturing(self) -> None:
        self._device_manager.stop(online=self._gaze_online)

    def start_capturing_microphone(self) -> None:
        folder = self.get_current_path()
        if folder is None:
            folder = self.session_path
            if folder is None:
                print("Cannot start microphone recording: session path not initialized.")
                return
        os.makedirs(folder, exist_ok=True)
        self.microphone.start_recording("recording.mp3", folder)

    def stop_capturing_microphone(self) -> None:
        self.microphone.stop_recording()

    def start_transcription(self, folder_path: Optional[str] = None) -> bool:
        folder = folder_path or self.get_current_path()
        if not folder:
            print("No folder available for transcription.")
            return False
        if not os.path.isdir(folder):
            print(f"Transcription folder does not exist: {folder}")
            return False
        return self.microphone.start_transcription(folder)

    def stop_transcription(self) -> None:
        self.microphone.stop_transcription()

    def set_whisper_mode(self, online: bool) -> None:
        self._whisper_online = online
        print(f"Whisper mode set to {'online' if online else 'offline'}")

    def set_gaze_mode(self, online: bool) -> None:
        self._gaze_online = online
        print(f"Gaze mode set to {'online' if online else 'offline'}")

    @property
    def whisper_online(self) -> bool:
        return self._whisper_online

    @property
    def gaze_online(self) -> bool:
        return self._gaze_online

    def log_token_placement(self, data: List[Any]) -> None:
        with open(os.path.join(self.current_path,"tokens_positions"), "w") as f:
            json.dump(data, f, indent=4)

    def record_tokenizer_info(self, tokenizer_name: str) -> None:
        if not tokenizer_name or not self.session_path:
            return
        if self.tokenizer_name == tokenizer_name:
            return
        metadata = {
            "tokenizer": tokenizer_name,
            "recorded_at": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()),
        }
        try:
            with open(os.path.join(self.session_path, "tokenizer_info.json"), "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=4)
            self.tokenizer_name = tokenizer_name
        except OSError as exc:
            print("Failed to write tokenizer metadata:", exc)

    def is_transcribing(self) -> bool:
        return self.microphone.is_transcribing()

    def get_current_session(self) -> Optional[str]:
        return self.session_path
    
    def get_current_path(self) -> Optional[str]:
        return self.current_path
