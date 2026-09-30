import os
import queue
import threading
import time

import sounddevice as sd
import soundfile as sf
from faster_whisper import WhisperModel
from PyQt5.QtCore import QThread, pyqtSignal
from typing import Optional

class Microphone:
    def __init__(self, sample_rate: Optional[int] = None, channels: int = 1, model_path: str = "models/small"):
        self.sample_rate = sample_rate
        if sample_rate is None:
            self.sample_rate = 44100
        self.channels = channels
        self.audio_q = queue.Queue()
        self.stop_event = threading.Event()

        sd.default.device = (2, None)
        self.filepath = None
        self.model_path = model_path
        self.model: Optional[WhisperModel] = None
        self.transcription_thread: Optional[threading.Thread] = None
        self.transcription_result: Optional[str] = None
        
        # self.model = self.setup_transcription()
    def set_device(self, device_number: int):
        sd.default.device = (device_number, None)
    
    def setup_transcription(self, model_path: str = "models/small") -> WhisperModel:
        model = WhisperModel(model_path, device = "cpu")
        return model

    def start_recording(self, filename, folder):
        os.makedirs(folder, exist_ok=True)
        self.filepath = os.path.join(folder, filename)
        self.stop_event.clear()
        self.thread_microphone = threading.Thread(target=self._record)
        self.thread_microphone.start()

    def _record(self):
        with sf.SoundFile(self.filepath, mode='w', samplerate = self.sample_rate, channels=self.channels) as file:
            with sd.InputStream(samplerate = self.sample_rate,
                                channels = self.channels,
                                callback = self._audio_callback):
                print("Recording")
                while not self.stop_event.is_set():
                    try:
                        data = self.audio_q.get(timeout=0.1)
                        file.write(data)
                    except queue.Empty:
                        pass
                print("Stopped recording")

    def stop_recording(self):
        self.stop_event.set()
        if self.thread_microphone is not None:
            self.thread_microphone.join()

    def start_transcription(self, folder: Optional[str]) -> bool:
        if not folder:
            print("start_transcription: folder is None")
            return False
        audio_file = os.path.join(folder, "recording.mp3")
        if not os.path.exists(audio_file):
            print(f"start_transcription: audio file not found: {audio_file}")
            return False
        if not self._ensure_model():
            print("start_transcription: whisper model unavailable")
            return False
        if self.transcription_thread and self.transcription_thread.is_alive():
            print("start_transcription: transcription already in progress")
            return False
        self.transcription_thread = threading.Thread(target=self._transcribe, args=(audio_file, folder))
        self.transcription_thread.start()
        return True

    def stop_transcription(self) -> None:
        if hasattr(self, "transcription_thread") and self.transcription_thread:
            self.transcription_thread.join(timeout=1.0)

    def _transcribe(self, audio_file: str, folder: str) -> None:
        print("Transcription started for", audio_file)
        try:
            segments, _ = self.model.transcribe(audio_file)
            transcript_lines = [segment.text.strip() for segment in segments if segment.text.strip()]
            transcript_text = "\n".join(transcript_lines)
            transcript_path = os.path.join(folder, "transcript.txt")
            with open(transcript_path, "w", encoding="utf-8") as f:
                f.write(transcript_text)
            self.transcription_result = transcript_text
            print("Transcription saved to:", transcript_path)
        except Exception as e:
            print("TRANSCRIBE ERROR:", e)
        finally:
            print("Transcription finished for", audio_file)
        

    def _audio_callback(self, indata, frames, time, status) -> None:
        if status:
            print("status: ", status)
        self.audio_q.put(indata.copy())

    def _ensure_model(self) -> bool:
        if self.model is not None:
            return True
        try:
            self.model = self.setup_transcription(self.model_path)
            return True
        except Exception as exc:
            print("Failed to load Whisper model:", exc)
            return False

    def is_transcribing(self) -> bool:
        return bool(self.transcription_thread and self.transcription_thread.is_alive())

class WhisperWorker(QThread):
    finished = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, model, filepath):
        super().__init__()
        self.model = model
        self.filepath = filepath
        self.model_path = "models/small"

    def run(self):
        try:
            segments, _ = self.model.transcribe(self.filepath)
            result = "\n".join([s.text for s in segments])
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))
  
