"""화면 색·글꼴 정의 (라이트/다크).

색은 토큰(이름 붙은 값)으로만 정의하고 스타일시트와 팔레트를 거기서 만든다. 글자와 배경의 대비는
라이트·다크 모두 WCAG 기준(본문 4.5:1 이상, 입력창 테두리 약 3:1)을 넘게 골랐다.
상태(완료/실패 등)는 색만으로 구분하지 않고 글자("● 완료")를 함께 쓴다.
"""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QGuiApplication, QPalette
from PySide6.QtWidgets import QApplication

# 한글은 맑은 고딕, 영문·숫자는 Segoe UI 순서로 쓴다 (Segoe UI에는 한글이 없어 자동으로 맑은 고딕으로 넘어간다)
FONT_STACK = "'Segoe UI', 'Malgun Gothic'"

LIGHT = {
    "bg": "#f3f5f4", "surface": "#ffffff", "field": "#f8faf8", "alt_row": "#f2f7f4",
    "border": "#d5dfd9", "field_border": "#7f988d", "text": "#1f3330", "muted": "#58696a",
    "title": "#2d5c4b", "accent": "#2b6f51", "accent_hover": "#215a41", "accent_soft": "#dceee3",
    "accent_text": "#1d6144", "chunk": "#3f9a72", "focus": "#2b6f51",
    "danger": "#a33b2a", "danger_soft": "#fbefeb", "hover": "#edf5ef", "disabled_text": "#7d8c86",
    "disabled_bg": "#eef2ef", "scroll": "#b9c9bf",
    "hero_bg": "#193f38", "hero_title": "#ffffff", "hero_desc": "#c5dbd5", "eyebrow": "#8fc2b2",
    "badge_bg": "#2b5148", "badge_text": "#d4eee5",
    "ok_fg": "#17623e", "ok_bg": "#e1f3e9", "fail_fg": "#a12c2c", "fail_bg": "#fbe6e3",
    "warn_fg": "#7a4f00", "warn_bg": "#fdf0d0", "info_fg": "#27567f", "info_bg": "#e2edf7",
    "idle_fg": "#58696a", "idle_bg": "#eceff0",
}

DARK = {
    "bg": "#0f1a18", "surface": "#172421", "field": "#1d2d2a", "alt_row": "#1b2b27",
    "border": "#2c423d", "field_border": "#5f7d73", "text": "#e4efeb", "muted": "#9fb3ac",
    "title": "#8fd6b5", "accent": "#5cc79a", "accent_hover": "#7ad9b1", "accent_soft": "#1f3a31",
    "accent_text": "#9fe3c4", "chunk": "#5cc79a", "focus": "#8fd6b5",
    "danger": "#ff9b8a", "danger_soft": "#3a2321", "hover": "#213631", "disabled_text": "#6f857d",
    "disabled_bg": "#1a2724", "scroll": "#3f5a52",
    "hero_bg": "#0d2a24", "hero_title": "#ffffff", "hero_desc": "#b5cfc8", "eyebrow": "#7dafa1",
    "badge_bg": "#1f3f37", "badge_text": "#c8ebdf",
    "ok_fg": "#7fe0b0", "ok_bg": "#143b2b", "fail_fg": "#ff9e9e", "fail_bg": "#4a1f22",
    "warn_fg": "#f5c86b", "warn_bg": "#43330f", "info_fg": "#9cc7f0", "info_bg": "#17304a",
    "idle_fg": "#9fb3ac", "idle_bg": "#22302d",
}

THEMES = {"light": LIGHT, "dark": DARK}
# 화면 모드 설정값: 자동(시스템 설정을 따름) / 라이트 / 다크
MODE_LABELS = {"auto": "자동", "light": "라이트", "dark": "다크"}
MODE_ORDER = ("auto", "light", "dark")

# 결과 표의 상태 글자 → (표시 글자, 색 종류)
STATUS_STYLE = {
    "완료": ("● 완료", "ok"),
    "일부 미번역": ("● 일부 미번역", "warn"),
    "검토 필요": ("● 검토 필요", "warn"),
    "실패": ("● 실패", "fail"),
    "취소됨": ("● 취소됨", "warn"),
    "대기": ("● 대기", "warn"),
    "대기열 추가": ("● 대기열", "info"),
    "변경 없음": ("● 변경 없음", "idle"),
}


def system_prefers_dark() -> bool:
    try:
        return QGuiApplication.styleHints().colorScheme() == Qt.ColorScheme.Dark
    except Exception:  # noqa: BLE001  (Qt 6.5 미만 등)
        return False


def resolve_mode(pref: str) -> str:
    """설정값(auto/light/dark)을 실제로 쓸 테마 이름(light/dark)으로 바꾼다."""
    if pref == "dark":
        return "dark"
    if pref == "light":
        return "light"
    return "dark" if system_prefers_dark() else "light"


def status_colors(mode: str, kind: str) -> tuple[QColor, QColor]:
    t = THEMES[mode]
    return QColor(t[f"{kind}_fg"]), QColor(t[f"{kind}_bg"])


def build_stylesheet(mode: str = "light") -> str:
    t = THEMES[mode]
    return f"""
QMainWindow, QWidget#root {{ background: {t['bg']}; }}
QWidget {{ font-family: {FONT_STACK}; font-size: 13px; color: {t['text']}; }}
QDialog, QMessageBox, QProgressDialog {{ background: {t['bg']}; }}
QToolTip {{ background: {t['surface']}; color: {t['text']}; border: 1px solid {t['border']}; padding: 4px 6px; }}
QWidget#hero {{ background: {t['hero_bg']}; border-radius: 18px; }}
QWidget#hero QLabel {{ background: transparent; }}
QLabel#heroTitle {{ color: {t['hero_title']}; font-size: 27px; font-weight: 700; }}
QLabel#heroDescription {{ color: {t['hero_desc']}; font-size: 13px; }}
QLabel#badge {{ color: {t['badge_text']}; background: {t['badge_bg']}; border-radius: 12px; padding: 8px 14px; }}
QLabel#eyebrow {{ color: {t['eyebrow']}; font-size: 11px; font-weight: 700; letter-spacing: 1px; }}
QPushButton#themeButton {{ background: {t['badge_bg']}; color: {t['badge_text']}; border: 1px solid transparent;
    border-radius: 12px; padding: 8px 14px; }}
QPushButton#themeButton:hover {{ border-color: {t['badge_text']}; }}
QLabel#muted {{ color: {t['muted']}; }}
QLabel#emptyState {{ color: {t['muted']}; background: {t['field']}; border: 1px dashed {t['field_border']};
    border-radius: 12px; padding: 22px; font-size: 14px; }}
QGroupBox {{ background: {t['surface']}; border: 1px solid {t['border']}; border-radius: 14px;
    margin-top: 22px; padding: 14px 12px 12px; }}
QGroupBox::title {{ subcontrol-origin: margin; left: 14px; top: 0px; padding: 0 4px;
    color: {t['title']}; font-size: 14px; font-weight: 700; }}
QScrollArea {{ border: none; background: transparent; }}
QWidget#settingsContent {{ background: transparent; }}
QListWidget, QPlainTextEdit, QTableWidget, QLineEdit {{ background: {t['field']};
    border: 1px solid {t['border']}; border-radius: 8px; padding: 6px;
    selection-background-color: {t['accent_soft']}; selection-color: {t['accent_text']}; }}
QLineEdit {{ border-color: {t['field_border']}; }}
QLineEdit:focus, QComboBox:focus, QListWidget:focus, QPlainTextEdit:focus, QTableWidget:focus {{
    border: 2px solid {t['focus']}; }}
QListWidget::item {{ padding: 8px; border-bottom: 1px solid {t['border']}; }}
QTableWidget {{ gridline-color: transparent; alternate-background-color: {t['alt_row']}; }}
QTableWidget::item {{ padding: 4px 8px; }}
QHeaderView::section {{ background: {t['accent_soft']}; border: none; padding: 8px 10px; color: {t['accent_text']};
    font-weight: 700; }}
QTabWidget::pane {{ border: none; background: {t['surface']}; }}
QTabBar::tab {{ background: {t['hover']}; color: {t['muted']}; border-radius: 8px;
    padding: 8px 16px; margin: 0 5px 8px 0; }}
QTabBar::tab:hover {{ color: {t['text']}; }}
QTabBar::tab:selected {{ background: {t['accent_soft']}; color: {t['accent_text']}; font-weight: 700; }}
QPushButton {{ background: {t['surface']}; border: 1px solid {t['field_border']}; border-radius: 9px;
    padding: 8px 13px; color: {t['text']}; }}
QPushButton:hover {{ background: {t['hover']}; border-color: {t['accent']}; }}
QPushButton:pressed {{ background: {t['accent_soft']}; }}
QPushButton:focus {{ border: 2px solid {t['focus']}; padding: 7px 12px; }}
QPushButton:disabled {{ color: {t['disabled_text']}; background: {t['disabled_bg']}; border-color: {t['border']}; }}
QPushButton#primaryButton {{ background: {t['accent']}; border-color: {t['accent']};
    color: {'#06231a' if mode == 'dark' else '#ffffff'}; font-size: 15px; font-weight: 700; padding: 12px 18px; }}
QPushButton#primaryButton:hover {{ background: {t['accent_hover']}; border-color: {t['accent_hover']}; }}
QPushButton#primaryButton:disabled {{ background: {t['disabled_bg']}; border-color: {t['border']}; color: {t['disabled_text']}; }}
QPushButton#dangerButton {{ color: {t['danger']}; border-color: {t['danger']}; }}
QPushButton#dangerButton:hover {{ background: {t['danger_soft']}; }}
QPushButton#dangerButton:disabled {{ color: {t['disabled_text']}; border-color: {t['border']}; }}
QComboBox {{ background: {t['field']}; border: 1px solid {t['field_border']}; border-radius: 8px;
    padding: 6px 10px; min-height: 22px; }}
QComboBox QAbstractItemView {{ background: {t['surface']}; color: {t['text']};
    selection-background-color: {t['accent_soft']}; selection-color: {t['accent_text']}; }}
QCheckBox {{ spacing: 8px; padding: 3px 0; }}
QProgressBar {{ background: {t['accent_soft']}; border: none; border-radius: 9px;
    min-height: 18px; text-align: center; color: {t['text']}; font-weight: 700; }}
QProgressBar::chunk {{ background: {t['chunk']}; border-radius: 9px; }}
QLabel#counts {{ color: {t['accent_text']}; font-weight: 600; }}
QLabel#statusText {{ font-size: 14px; font-weight: 600; }}
QLabel#timeLabel {{ color: {t['muted']}; font-size: 13px; padding: 2px 0 4px; }}
QScrollBar:vertical {{ background: transparent; width: 10px; margin: 2px; }}
QScrollBar::handle:vertical {{ background: {t['scroll']}; border-radius: 4px; min-height: 30px; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0px; }}
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{ background: transparent; }}
QScrollBar:horizontal {{ background: transparent; height: 10px; margin: 2px; }}
QScrollBar::handle:horizontal {{ background: {t['scroll']}; border-radius: 4px; min-width: 30px; }}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{ width: 0px; }}
QSplitter::handle {{ background: transparent; width: 14px; }}
"""


def build_palette(mode: str) -> QPalette:
    """Fusion 스타일이 체크박스·콤보박스 화살표·메시지 상자 등을 이 팔레트로 그리게 한다."""
    t = THEMES[mode]
    p = QPalette()
    roles = {
        QPalette.Window: t["bg"], QPalette.WindowText: t["text"], QPalette.Base: t["field"],
        QPalette.AlternateBase: t["alt_row"], QPalette.Text: t["text"], QPalette.Button: t["surface"],
        QPalette.ButtonText: t["text"], QPalette.ToolTipBase: t["surface"], QPalette.ToolTipText: t["text"],
        QPalette.Highlight: t["accent"], QPalette.HighlightedText: "#06231a" if mode == "dark" else "#ffffff",
        QPalette.PlaceholderText: t["muted"], QPalette.Link: t["accent_text"], QPalette.BrightText: "#ffffff",
    }
    for role, color in roles.items():
        p.setColor(role, QColor(color))
    for role in (QPalette.WindowText, QPalette.Text, QPalette.ButtonText):
        p.setColor(QPalette.Disabled, role, QColor(t["disabled_text"]))
    return p


def apply_theme(mode: str) -> None:
    """앱 전체에 테마를 적용한다. (스타일시트는 앱에 걸어서 모든 대화상자에도 똑같이 먹는다)"""
    app = QApplication.instance()
    if app is None:
        return
    if app.style().objectName().lower() != "fusion":
        app.setStyle("Fusion")
    app.setPalette(build_palette(mode))
    app.setStyleSheet(build_stylesheet(mode))


STYLE_SHEET = build_stylesheet("light")  # 예전 코드 호환용
