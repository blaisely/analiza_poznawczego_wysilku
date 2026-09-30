from typing import Final

from PyQt5.QtWidgets import QPushButton, QWidget


APP_STYLESHEET: Final[str] = """
QMainWindow, QDialog, QWidget {
    background-color: #f4efe7;
    color: #243b53;
    font-family: "Montserrat Medium";
    font-size: 15px;
}

QLabel {
    background: transparent;
    color: #243b53;
}

QLabel#main_title, QLabel#end_title, QLabel#settings_title {
    color: #102a43;
    font-size: 30px;
    font-weight: 700;
}

QLabel#main_subtitle, QLabel#end_subtitle, QLabel#settings_subtitle {
    color: #52606d;
    font-size: 15px;
}

QLabel#question_label {
    color: #486581;
    font-size: 22px;
    font-weight: 600;
    letter-spacing: 0.08em;
}

QGroupBox,
QFrame#summary_frame,
QFrame#settings_card {
    background-color: #fffdf9;
    border: 1px solid #d9e2ec;
    border-radius: 24px;
}

QGroupBox {
    margin-top: 18px;
    padding: 24px;
    font-size: 18px;
    font-weight: 600;
    color: #102a43;
}

QGroupBox::title {
    subcontrol-origin: margin;
    left: 24px;
    padding: 0 8px;
}

QTextBrowser#text_browser,
QTextEdit,
QComboBox,
QSpinBox {
    background-color: #fffaf2;
    border: 1px solid #cbd5e1;
    border-radius: 18px;
    color: #102a43;
}

QTextBrowser#text_browser,
QTextEdit {
    padding: 18px 22px;
}

QComboBox,
QSpinBox {
    padding: 10px 14px;
    min-height: 28px;
}

QComboBox::drop-down,
QSpinBox::up-button,
QSpinBox::down-button {
    border: none;
    width: 28px;
}

QPushButton {
    min-height: 28px;
    border: 1px solid transparent;
    border-radius: 18px;
    padding: 12px 22px;
    font-size: 16px;
    font-weight: 600;
    background-color: #d9e2ec;
    color: #102a43;
}

QPushButton:hover {
    background-color: #cbd5e1;
}

QPushButton:pressed {
    background-color: #bcccdc;
}

QPushButton:disabled {
    background-color: #d9e2ec;
    color: #829ab1;
    border-color: #d9e2ec;
}

QPushButton[variant="primary"] {
    background-color: #2f6fed;
    color: #ffffff;
}

QPushButton[variant="primary"]:hover {
    background-color: #1d5ed8;
}

QPushButton[variant="primary"]:pressed {
    background-color: #174bb0;
}

QPushButton[variant="secondary"] {
    background-color: #e4edf5;
    color: #1f3a56;
    border-color: #d9e2ec;
}

QPushButton[variant="success"] {
    background-color: #0f766e;
    color: #ffffff;
}

QPushButton[variant="success"]:hover {
    background-color: #0b5f58;
}

QPushButton[variant="recording"] {
    background-color: #b42318;
    color: #ffffff;
}

QPushButton[variant="recording"]:hover {
    background-color: #981b12;
}

QSlider::groove:horizontal {
    height: 8px;
    background: #d9e2ec;
    border-radius: 4px;
}

QSlider::sub-page:horizontal {
    background: #2f6fed;
    border-radius: 4px;
}

QSlider::handle:horizontal {
    width: 20px;
    margin: -7px 0;
    border-radius: 10px;
    background: #ffffff;
    border: 2px solid #2f6fed;
}
"""


def apply_theme(widget: QWidget) -> None:
    widget.setStyleSheet(APP_STYLESHEET)


def set_button_variant(button: QPushButton, variant: str) -> None:
    button.setProperty("variant", variant)
    refresh_widget_style(button)


def refresh_widget_style(widget: QWidget) -> None:
    style = widget.style()
    style.unpolish(widget)
    style.polish(widget)
    widget.update()
