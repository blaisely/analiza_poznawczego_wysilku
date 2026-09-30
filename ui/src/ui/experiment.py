import sys
import traceback
from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel, QWidget, 
                             QVBoxLayout, QHBoxLayout, QGridLayout, QPushButton, QMessageBox, QDialog,
                             QTextBrowser, QSlider, QStyle)
from PyQt5 import uic
from PyQt5.QtGui import QTextCursor, QTextBlockFormat, QTextOption, QPalette, QCloseEvent
from PyQt5.QtCore import QRectF
import os, string, json, time, hashlib
import math
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QPainter, QColor, QPen
from PyQt5.QtCore import Qt
import re
from typing import Any, cast, List, Optional, Tuple
from session_manager import SessionManager
from window_manager import WindowManager
from settings import SettingsDialog
from theme import apply_theme, set_button_variant
try:
    from transformers import AutoTokenizer
except ImportError:
    AutoTokenizer = None  # type: ignore[assignment]

_TOKENIZER: Optional[Any] = None
_TOKENIZER_MODEL_NAME = os.environ.get("MAGISTERKA_TOKENIZER", "bert-base-multilingual-cased")
DEFAULT_TEXT_POINT_SIZE = 15
DEFAULT_TEXT_LINE_HEIGHT_PERCENT = 125

def get_tokenizer_instance() -> Optional[Any]:
    global _TOKENIZER
    if _TOKENIZER is not None:
        return _TOKENIZER
    if AutoTokenizer is None:
        print("Transformers not available; falling back to word-based tokenization.")
        return None
    try:
        tokenizer = AutoTokenizer.from_pretrained(_TOKENIZER_MODEL_NAME, use_fast=True)
        _TOKENIZER = tokenizer
        print(f"Loaded tokenizer {_TOKENIZER_MODEL_NAME}")
        return tokenizer
    except Exception as exc:
        print(f"Unable to load tokenizer {_TOKENIZER_MODEL_NAME}: {exc}")
        return None

class Window():
    def add_window_manager(self, window_manager: WindowManager):
        raise NotImplemented
    def add_session_manager(self, session_manager: SessionManager):
        raise NotImplemented
    def add_app(self,app):
        raise NotImplemented
    def init(self):
        raise NotImplemented

class ProcessingDialog(QDialog):
    def __init__(self, title: str, message: str, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._allow_close = False
        self.setWindowTitle(title)
        self.setWindowModality(Qt.ApplicationModal)
        self.setModal(True)
        self.setWindowFlags(Qt.Dialog | Qt.CustomizeWindowHint | Qt.WindowTitleHint)
        self.setFixedSize(360, 120)
        layout = QVBoxLayout(self)
        label = QLabel(message, self)
        label.setWordWrap(True)
        label.setAlignment(Qt.AlignCenter)
        layout.addWidget(label)
        apply_theme(self)

    def safe_close(self) -> None:
        self._allow_close = True
        self.close()

    def closeEvent(self, event: QCloseEvent) -> None:
        if self._allow_close:
            event.accept()
            return
        event.ignore()

    def keyPressEvent(self, event) -> None:
        if not self._allow_close and event.key() == Qt.Key_Escape:
            event.ignore()
            return
        super().keyPressEvent(event)

    def mousePressEvent(self, event) -> None:
        if not self._allow_close:
            event.ignore()
            return
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if not self._allow_close:
            event.ignore()
            return
        super().mouseReleaseEvent(event)


class SnapSlider(QSlider):
    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.LeftButton and self.orientation() == Qt.Horizontal:
            value = QStyle.sliderValueFromPosition(
                self.minimum(),
                self.maximum(),
                event.x(),
                max(self.width(), 1),
                self.invertedAppearance(),
            )
            self.setValue(value)
            event.accept()
        super().mousePressEvent(event)


class ExperimentWindow(QMainWindow, Window):
    def __init__(self):
        super().__init__()
        uic.loadUi("ui/experiment_ui.ui", self)
        apply_theme(self)
        self.samples_per_language = 6
        text_paths = self._collect_text_paths("questions")
        self.qa_samples = self.load_qa_samples()
        self.generator = self._mixed_generator(text_paths, self.qa_samples)
        self.n_texts = len(text_paths) + len(self.qa_samples)
        self.current_question = 0
        self.text_point_size = self._read_env_int("MAGISTERKA_TEXT_POINT_SIZE", DEFAULT_TEXT_POINT_SIZE)
        self.text_line_height_percent = self._read_env_int("MAGISTERKA_TEXT_LINE_HEIGHT", DEFAULT_TEXT_LINE_HEIGHT_PERCENT)
        self.next_button.clicked.connect(self.on_click) 
        self.mic_button.clicked.connect(self.on_click_microphone)
        self.mic_on = False
        self.gaze_dialog = ProcessingDialog("Przetwarzanie", "Trwa przetwarzanie danych z eyetrackera...", self)
        self.audio_dialog = ProcessingDialog("Transkrypcja", "Trwa transkrypcja audio...", self)
        self.processing_active = False
        self.transcription_poll_timer = QTimer(self)
        self.transcription_poll_timer.setInterval(200)
        self.transcription_poll_timer.timeout.connect(self._poll_transcription_status)
        self._configure_visuals()

    def init(self):
        """
        Lazy initialization
        """
        QTimer.singleShot(50, self._delayed_init)

    
    def _collect_text_paths(self, base_path: str) -> list[str]:
        """
        Gather a deterministic list of text file paths:
        - 6 Polish files from questions/pl (prefer longer texts)
        - 6 English files from questions/eng
        """
        paths: list[str] = []
        pl_dir = os.path.join(base_path, "pl")
        en_dir = os.path.join(base_path, "eng")

        if os.path.isdir(pl_dir):
            pl_files = sorted(f for f in os.listdir(pl_dir) if f.lower().endswith(".txt"))
            pl_with_length: list[tuple[str, int]] = []
            for file_name in pl_files:
                full_path = os.path.join(pl_dir, file_name)
                try:
                    length = len(self._read_text_file(full_path))
                except OSError:
                    length = 0
                pl_with_length.append((file_name, length))
            pl_with_length.sort(key=lambda x: (-x[1], x[0]))
            selected_pl = [file_name for file_name, _ in pl_with_length[: self.samples_per_language]]
            paths.extend(os.path.join(pl_dir, file_name) for file_name in selected_pl)
        else:
            print(f"Missing questions directory: {pl_dir}")

        if os.path.isdir(en_dir):
            en_files = sorted(f for f in os.listdir(en_dir) if f.lower().endswith(".txt"))
            selected_en = en_files[: self.samples_per_language]
            paths.extend(os.path.join(en_dir, file_name) for file_name in selected_en)
        else:
            print(f"Missing questions directory: {en_dir}")
        return paths

    def load_common_knowledge(self) -> list[dict[str, str]]:
        """
        Load deterministic curated QA samples:
        - 6 Polish entries (prefer longer passages/questions)
        - 6 English entries
        """
        qa_path = os.path.join("questions", "q&a", "common_knowledge.json")
        if not os.path.exists(qa_path):
            print(f"Common knowledge file missing: {qa_path}")
            return []
        try:
            with open(qa_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as exc:
            print(f"Unable to load QA samples from {qa_path}: {exc}")
            return []
        if not isinstance(data, list):
            print(f"Unexpected QA format in {qa_path}")
            return []
        samples = [item for item in data if isinstance(item, dict)]
        pl_samples = [item for item in samples if str(item.get("lang", "")).strip().lower() == "pl"]
        en_samples = [item for item in samples if str(item.get("lang", "")).strip().lower() == "en"]
        pl_samples.sort(
            key=lambda item: (
                -(len(str(item.get("passage", ""))) + len(str(item.get("question", "")))),
                str(item.get("id", "")),
            )
        )
        selected_pl = pl_samples[: self.samples_per_language]
        selected_en = en_samples[: self.samples_per_language]
        return selected_pl + selected_en

    def load_qa_samples(self) -> list[dict[str, str]]:
        """
        Load QA data based on MAGISTERKA_QA_MODE.
        Modes: none | common | all
        """
        mode = os.environ.get("MAGISTERKA_QA_MODE", "common").strip().lower()
        if mode in {"none", "off", "0", "false"}:
            return []

        samples: list[dict[str, str]] = []
        if mode in {"common", "all", "full"}:
            samples.extend(self.load_common_knowledge())

        return self._apply_sample_limit(samples, "MAGISTERKA_QA_LIMIT")

    def _apply_sample_limit(
        self,
        samples: list[dict[str, str]],
        env_var: str,
    ) -> list[dict[str, str]]:
        limit = self._read_env_int(env_var, len(samples))
        if limit > 0 and len(samples) > limit:
            return samples[:limit]
        return samples

    def _read_env_int(self, env_var: str, default: int) -> int:
        raw = os.environ.get(env_var, "").strip()
        if not raw:
            return default
        try:
            return int(raw)
        except ValueError:
            return default

    def _mixed_generator(self, text_paths: list[str], qa_samples: list[dict[str, str]]):
        """
        Yield text passages and QA samples in a mixed order, inserting QA entries evenly.
        """
        total_text = len(text_paths)
        total_qa = len(qa_samples)
        interval = math.ceil(total_text / total_qa) if total_text and total_qa else None
        qa_index = 0
        text_count = 0

        for path in text_paths:
            try:
                text = self._read_text_file(path)
            except Exception as exc:
                print(f"Unable to read {path}: {exc}")
                continue
            text_count += 1
            yield {"type": "text", "text": text}
            if interval and qa_index < total_qa and text_count >= interval * (qa_index + 1):
                yield self._format_qa_sample(qa_samples[qa_index])
                qa_index += 1

        while qa_index < total_qa:
            yield self._format_qa_sample(qa_samples[qa_index])
            qa_index += 1

    def _format_qa_sample(self, qa_sample: dict[str, str]) -> dict[str, str]:
        """
        Normalize QA metadata so the UI can treat it like a text sample with extra fields.
        """
        return {
            "type": "qa",
            "passage": qa_sample.get("passage", "").strip(),
            "question": qa_sample.get("question", "").strip(),
            "answer": qa_sample.get("answer", "")
        }

    def _read_text_file(self, path: str) -> str:
        with open(path, "r", encoding="utf-8") as f:
            return self._clean_text(f.read())

    def _clean_text(self, text: str) -> str:
        if not text:
            return ""
        return "\n".join(line.rstrip() for line in text.splitlines())


    def _delayed_init(self) -> None:
        """
        Delayed initilization of the UI
        """
        self.current_question = self.current_question + 1
        folder = self.session_manager.create_question_folder()
        if not folder:
            QMessageBox.warning(self, "Błąd sesji", "Nie udało się utworzyć folderu dla bieżącego pytania.")
            self.window_manager.change_window("end")
            return
        try:
            sentence = self.load_text()
        except StopIteration:
            print("Wszystkie teksty zostały zakończone - przejście do okna końcowego.")
            self.window_manager.change_window("end")
            return
        
        self.session_manager.start_capturing()
        
        text_browser = cast(QTextBrowser, self.text_browser)
        self._set_experiment_text(text_browser, sentence)
        
        self.app.processEvents()
        rects = self.get_token_rects(text_browser)
        self.tokens = self.rect_to_dict(rects)

        # might add self. before?
        text_browser.viewport().update()
        self.next_button.setEnabled(True)
        self._restore_experiment_focus()


    def rect_to_dict(self, rects: list):
        """
        Convert PyQT Rectangle coordinates in a list of dictionaries
        """
        token_list = []
        for token, rect in rects:
            coord_dict = {}
            coord_dict["x"] = rect.x()
            coord_dict["y"] = rect.y()
            coord_dict["width"] = rect.width()
            coord_dict["height"] = rect.height()
            token_list.append((token, coord_dict))
            # print(token_list)
        return token_list

    def load_text(self) -> str:
        """
        Load text into the UI
        """
        sample = next(self.generator)
        sample_type = sample.get("type")
        label_prefix = "Pytanie" if sample_type == "qa" else "Tekst"
        self.question_label.setText(f"{label_prefix} {self.current_question}")

        if sample_type == "qa":
            question = self._clean_text(sample.get("question", ""))
            passage = self._clean_text(sample.get("passage", ""))
            parts: list[str] = []
            parts.append("Kontekst:")
            if passage:
                parts.append(passage)
            parts.append("")
            parts.append("Pytanie:")
            if question:
                parts.append(question)
            return "\n".join(parts)

        return self._clean_text(sample.get("text", ""))

    def _configure_visuals(self) -> None:
        self.question_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.text_browser.document().setDocumentMargin(0.0)
        self.text_browser.setOpenExternalLinks(False)
        self.text_browser.setOpenLinks(False)
        self.text_browser.setUndoRedoEnabled(False)
        self.text_browser.setReadOnly(True)
        font = self.text_browser.font()
        font.setPointSize(self.text_point_size)
        self.text_browser.setFont(font)
        self.text_browser.document().setDefaultFont(font)
        self.text_browser.setStyleSheet(
            f"font-size: {self.text_point_size}pt; line-height: {self.text_line_height_percent}%;"
        )
        self._set_mic_button_idle()
        set_button_variant(self.next_button, "primary")

    def _set_experiment_text(self, text_browser: QTextBrowser, text: str) -> None:
        text_browser.setPlainText(text)
        text_browser.moveCursor(QTextCursor.Start)
        doc = text_browser.document()
        doc.setDocumentMargin(0.0)
        text_option = doc.defaultTextOption()
        text_option.setAlignment(Qt.AlignLeft)
        text_option.setWrapMode(QTextOption.WrapAtWordBoundaryOrAnywhere)
        doc.setDefaultTextOption(text_option)

        cursor = QTextCursor(doc)
        cursor.select(QTextCursor.Document)
        block_format = QTextBlockFormat()
        block_format.setAlignment(Qt.AlignLeft)
        block_format.setLineHeight(self.text_line_height_percent, QTextBlockFormat.ProportionalHeight)
        block_format.setTopMargin(0.0)
        block_format.setBottomMargin(14.0)
        cursor.mergeBlockFormat(block_format)
        cursor.clearSelection()
        text_browser.setTextCursor(cursor)
        text_browser.moveCursor(QTextCursor.Start)

    def _set_mic_button_idle(self) -> None:
        self.mic_button.setText("NAGRAJ ODPOWIEDZ")
        set_button_variant(self.mic_button, "success")

    def _set_mic_button_recording(self) -> None:
        self.mic_button.setText("ZATRZYMAJ NAGRYWANIE")
        set_button_variant(self.mic_button, "recording")

    def _close_processing_dialog(self, dialog: ProcessingDialog) -> None:
        dialog.safe_close()
        dialog.hide()
        self.app.processEvents()

    def _restore_experiment_focus(self) -> None:
        self.raise_()
        self.activateWindow()
        self.setFocus(Qt.OtherFocusReason)
        self.next_button.setFocus(Qt.OtherFocusReason)
        self.app.processEvents()
    
    def add_window_manager(self, window_manager : WindowManager):
        self.window_manager = window_manager

    def add_session_manager(self, session_manager : SessionManager):
        self.session_manager = session_manager

    def add_app(self,app):
        self.app = app

    def get_token_rects(self, text_browser: QTextBrowser) -> List[Tuple[str, QRectF]]:
        """Compute (token, QRectF) for each tokenizer token in the QTextBrowser."""
        doc = text_browser.document()
        full_text = doc.toPlainText()
        # tokenizer = get_tokenizer_instance()
        tokenizer = None
        if tokenizer is None:
            return self.get_word_rects(text_browser)

        try:
            encoding = tokenizer(
                full_text,
                return_offsets_mapping=True,
                add_special_tokens=False,
                truncation=False,
            )
        except Exception as exc:
            print("Tokenization failed, falling back to word rectangles:", exc)
            return self.get_word_rects(text_browser)

        token_strings = encoding.tokens()
        offsets = encoding.get("offset_mapping")
        if not offsets or len(offsets) != len(token_strings):
            raise ValueError("Unexpected offset map from tokenizer.")
        rects: List[Tuple[str, QRectF]] = []
        cursor = QTextCursor(doc)

        for token, offset in zip(token_strings, offsets):
            start, end = offset
            if start == end:
                continue
            cursor.setPosition(start)
            start_rect = text_browser.cursorRect(cursor)
            cursor.setPosition(max(start, end))
            end_rect = text_browser.cursorRect(cursor)

            global_start = text_browser.mapToGlobal(start_rect.topLeft())
            global_end = text_browser.mapToGlobal(end_rect.bottomRight())

            width = max(global_end.x() - global_start.x(), 1.0)
            height = max(global_end.y() - global_start.y(), 1.0)
            rect = QRectF(global_start.x(), global_start.y(), width, height)
            rects.append((token, rect))

        text_browser.moveCursor(QTextCursor.Start)
        if not rects:
            return self.get_word_rects(text_browser)

        tokenizer_name = getattr(tokenizer, "name_or_path", _TOKENIZER_MODEL_NAME)
        if self.session_manager:
            self.session_manager.record_tokenizer_info(tokenizer_name)

        return rects

    def get_word_rects(self, text_browser: QTextBrowser) -> List[Tuple[str, QRectF]]:
        """Fall back to word rectangles when tokenizer offsets are unavailable."""
        doc = text_browser.document()
        token_pattern = re.compile(r"\w+(-\w+)*|[^\w\s]", re.UNICODE)
        full_text = doc.toPlainText()

        rects: List[Tuple[str, QRectF]] = []
        cursor = QTextCursor(doc)

        for match in token_pattern.finditer(full_text):
            start = match.start()
            end = match.end()
            token = match.group()

            cursor.setPosition(start)
            start_rect = text_browser.cursorRect(cursor)
            cursor.setPosition(end)
            end_rect = text_browser.cursorRect(cursor)

            global_start_pos = text_browser.mapToGlobal(start_rect.topLeft())
            global_end_pos = text_browser.mapToGlobal(end_rect.bottomRight())

            width = max(global_end_pos.x() - global_start_pos.x(), 1.0)
            height = max(global_end_pos.y() - global_start_pos.y(), 1.0)
            global_rect = QRectF(global_start_pos.x(), global_start_pos.y(), width, height)
            rects.append((token, global_rect))

        text_browser.moveCursor(QTextCursor.Start)
        return rects

    def closeEvent(self, event: QCloseEvent) -> None:
        if self.processing_active:
            QMessageBox.warning(self, "Przetwarzanie w toku", "Poczekaj na zakończenie przetwarzania przed zamknięciem.")
            event.ignore()
        elif self.session_manager and self.session_manager.is_transcribing():
            QMessageBox.warning(self, "Transkrypcja w toku", "Poczekaj na zakończenie transkrypcji przed zamknięciem.")
            event.ignore()
        else:
            event.accept()
        super().closeEvent(event)
    # TODO: IMplement version for tokens Bert/OpenAI etc.
    
    # def get_token_rects(self, text_browser: QTextBrowser):
    #     """Compute (token, QRectF) for each tiktoken token in the QTextBrowser."""
    #     doc = text_browser.document()
    #     full_text = doc.toPlainText()

    #     # --- PRECOMPUTE TOKENS ---
    #     token_offsets = get_tiktoken_offsets(full_text)  # [(token, start, end), ...]

    #     rects = []
    #     cursor = QTextCursor(doc)

    #     for token, start, end in token_offsets:
    #         cursor.setPosition(start)
    #         start_rect = text_browser.cursorRect(cursor)

    #         cursor.setPosition(end)
    #         end_rect = text_browser.cursorRect(cursor)

    #         # Bounding rect in widget coordinates
    #         rect = QRectF(
    #             start_rect.x(),
    #             start_rect.y(),
    #             max(end_rect.x() - start_rect.x(), 1.0),
    #             max(start_rect.height(), end_rect.height())
    #         )

    #         # Convert to global screen coordinates
    #         global_start = text_browser.mapToGlobal(start_rect.topLeft())
    #         global_end = text_browser.mapToGlobal(end_rect.bottomRight())
    #         global_rect = QRectF(
    #             global_start.x(),
    #             global_start.y(),
    #             global_end.x() - global_start.x(),
    #             global_end.y() - global_start.y()
    #         )

    #         rects.append((token, global_rect))

    #     # Reset cursor to start
    #     text_browser.moveCursor(QTextCursor.Start)

    #     return rects

    def start_transcription(self) -> None:
        folder = self.session_manager.get_current_path()
        if not folder:
            print("No current folder available for transcription.")
            self._finish_processing_pipeline()
            return
        if not self.session_manager.whisper_online:
            print("Whisper offline mode - transcription deferred.")
            self._finish_processing_pipeline()
            return
        self.audio_dialog.show()
        self.app.processEvents()
        started = self.session_manager.start_transcription(folder)
        if not started:
            self._close_processing_dialog(self.audio_dialog)
            self._finish_processing_pipeline()
            return
        self.transcription_poll_timer.start()

    def on_click(self):
        """
        Next button callback

        Stop recording and log token placement
        """
        if self.processing_active:
            return
        self.processing_active = True
        self.next_button.setEnabled(False)
        self.session_manager.log_token_placement(self.tokens)
        self._run_gaze_processing()

    def _run_gaze_processing(self) -> None:
        self.gaze_dialog.show()
        self.app.processEvents()
        try:
            self.session_manager.stop_capturing()
        except Exception as exc:
            error_details = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
            self._on_gaze_processing_error(error_details)
            return
        self._on_gaze_processing_finished()

    def _on_gaze_processing_finished(self) -> None:
        self._close_processing_dialog(self.gaze_dialog)
        self.start_transcription()

    def _on_gaze_processing_error(self, error_message: str) -> None:
        print("Przetwarzanie danych wzroku nie powiodło się:", error_message)
        self._close_processing_dialog(self.gaze_dialog)
        QMessageBox.critical(self, "Błąd przetwarzania danych wzroku", error_message)
        self.start_transcription()

    def _poll_transcription_status(self) -> None:
        if self.session_manager and self.session_manager.is_transcribing():
            return
        self.transcription_poll_timer.stop()
        self._close_processing_dialog(self.audio_dialog)
        self._finish_processing_pipeline()

    def _finish_processing_pipeline(self) -> None:
        self.processing_active = False
        self.next_button.setEnabled(True)
        self.init()

    def on_click_microphone(self):
        """
        Microphone button callback

        Start/Stop recording, save audio
        """
        self.mic_on = not self.mic_on
        if(self.mic_on):
            self.session_manager.start_capturing_microphone()
            self._set_mic_button_recording()
        else:
            self.session_manager.stop_capturing_microphone()
            self._set_mic_button_idle()
            
            time.sleep(0.5)
            
class MainWindow(QMainWindow, Window):
    def __init__(self):
        super().__init__()
        uic.loadUi("ui/main_ui.ui", self)
        apply_theme(self)
        self._setup_sliders()
        self.next_button.clicked.connect(self.on_click)
        self.settings_button.clicked.connect(self.settings_button_on_click)
        set_button_variant(self.next_button, "primary")
        set_button_variant(self.settings_button, "secondary")
        
    def init(self):
        pass

    def _setup_sliders(self) -> None:
        slider_specs: list[tuple[str, int]] = [
            ("speaking_box", 3),
            ("reading_box", 4),
            ("horizontalSlider", 5),
            ("horizontalSlider_2", 6),
        ]
        grid_layout = cast(QGridLayout, self.groupBox.layout())
        for slider_name, row in slider_specs:
            original_slider = cast(QSlider, getattr(self, slider_name))
            slider = self._replace_slider_with_snap_slider(original_slider)
            setattr(self, slider_name, slider)

            value_label = QLabel(str(slider.value()), self.groupBox)
            value_label.setObjectName(f"{slider_name}_value_label")
            value_label.setAlignment(Qt.AlignCenter)
            value_label.setMinimumWidth(44)
            grid_layout.addWidget(value_label, row, 2)
            slider.valueChanged.connect(
                lambda value, label=value_label, current_slider=slider: self._update_slider_value_label(
                    current_slider,
                    label,
                    value,
                )
            )
            self._update_slider_value_label(slider, value_label, slider.value())

        grid_layout.setColumnStretch(0, 2)
        grid_layout.setColumnStretch(1, 5)
        grid_layout.setColumnStretch(2, 0)

    def _replace_slider_with_snap_slider(self, original_slider: QSlider) -> QSlider:
        slider = SnapSlider(original_slider.parentWidget())
        slider.setObjectName(original_slider.objectName())
        slider.setMinimum(original_slider.minimum())
        slider.setMaximum(original_slider.maximum())
        slider.setSingleStep(original_slider.singleStep())
        slider.setPageStep(1)
        slider.setValue(original_slider.value())
        slider.setOrientation(original_slider.orientation())
        slider.setTickPosition(original_slider.tickPosition())
        slider.setTickInterval(original_slider.tickInterval())
        slider.setMinimumSize(original_slider.minimumSize())
        slider.setMaximumSize(original_slider.maximumSize())
        slider.setSizePolicy(original_slider.sizePolicy())
        slider.setInvertedAppearance(original_slider.invertedAppearance())
        slider.setInvertedControls(original_slider.invertedControls())

        grid_layout = cast(QGridLayout, self.groupBox.layout())
        row, column, row_span, column_span = grid_layout.getItemPosition(grid_layout.indexOf(original_slider))
        grid_layout.removeWidget(original_slider)
        original_slider.deleteLater()
        grid_layout.addWidget(slider, row, column, row_span, column_span)
        return slider

    def _update_slider_value_label(self, slider: QSlider, label: QLabel, value: int) -> None:
        label.setText(str(value))
        slider.setToolTip(str(value))

    def add_window_manager(self, window_manager: WindowManager):
        self.window_manager = window_manager

    def add_session_manager(self, session_manager: SessionManager):
        self.session_manager = session_manager

    def add_app(self,app):
        self.app = app

    def settings_button_on_click(self):
        dialog = QDialog()
        settings_dialog = SettingsDialog(self.session_manager)
        settings_dialog.setupUi(dialog)
        dialog.exec_()

    def on_click(self):
        participant_data: dict[str, Any] = {}
        participant_data["age"]  = self.age_box.value()
        participant_data["L1"] = self.language_box.currentText()
        participant_data["L2_level"] = self.level_box.currentText()
        participant_data["reading_level"] = self.reading_box.value()
        participant_data["speaking_level"] = self.speaking_box.value()
        participant_data["tiredness_level"] = self.horizontalSlider.value()
        participant_data["stress_level"] = self.horizontalSlider_2.value()
        participant_data["state_notes"] = self.state_notes_input.toPlainText().strip()
        if self.session_manager is not None:
            self.session_manager.start_new_session(participant_data)
            if hasattr(self, "window_manager"):
                self.window_manager.next_window()
        else:
            print("Session manager not set")

    def closeEvent(self, event: QCloseEvent) -> None:
        if self.session_manager and self.session_manager.is_transcribing():
            QMessageBox.warning(self, "Transkrypcja w toku", "Poczekaj na zakończenie transkrypcji przed zamknięciem.")
            event.ignore()
        else:
            event.accept()
        super().closeEvent(event)

    def keyPressEvent(self, a0):
        if a0.key() == Qt.Key.Key_Escape and self.isWindow():
            if self.isFullScreen():
                self.showNormal()
            else:
                self.showFullScreen()
            return
        super().keyPressEvent(a0)


class EndWindow(QMainWindow, Window):
    def __init__(self):
        super().__init__()
        uic.loadUi("ui/end_ui.ui", self)
        apply_theme(self)
        layout = cast(QVBoxLayout, self.centralwidget.layout())
        layout.setAlignment(self.summary_frame, Qt.AlignHCenter)
    def add_window_manager(self, window_manager):
        self.window_manager = window_manager
    def add_session_manager(self, session_manager):
        self.session_manager = session_manager
    def add_app(self,app):
        self.app = app
    def init(self):
        self.app.processEvents()
