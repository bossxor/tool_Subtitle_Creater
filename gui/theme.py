"""Shared colors and typography for the desktop workspace."""

STYLE_SHEET = """
QMainWindow, QWidget#root { background: #f3f5f4; }
QWidget { font-family: 'Malgun Gothic'; font-size: 12px; color: #263b38; }
QWidget#hero { background: #193f38; border-radius: 18px; }
QLabel#heroTitle { color: #ffffff; font-size: 27px; font-weight: 700; }
QLabel#heroDescription { color: #c5dbd5; font-size: 12px; }
QLabel#badge { color: #d4eee5; background: #2b5148; border-radius: 12px; padding: 8px 14px; }
QLabel#eyebrow { color: #7dafa1; font-size: 11px; font-weight: 700; }
QLabel#muted { color: #758781; }
QLabel#emptyState { color: #7c9088; background: #f1f7f3; border: 1px dashed #bcd2c7;
    border-radius: 12px; padding: 22px; font-size: 13px; }
QGroupBox { background: #ffffff; border: 1px solid #e1e8e3; border-radius: 14px;
    margin-top: 22px; padding: 14px 12px 12px; }
QGroupBox::title { subcontrol-origin: margin; left: 14px; top: 0px; padding: 0 4px;
    color: #3d6457; font-size: 13px; font-weight: 700; }
QScrollArea { border: none; background: transparent; }
QWidget#settingsContent { background: transparent; }
QListWidget, QPlainTextEdit, QTableWidget, QLineEdit { background: #f8faf8;
    border: 1px solid #e0e8e2; border-radius: 8px; padding: 6px;
    selection-background-color: #d9eee3; selection-color: #193f38; }
QLineEdit:focus, QComboBox:focus, QListWidget:focus { border-color: #579d7c; }
QListWidget::item { padding: 8px; border-bottom: 1px solid #e9eeea; }
QHeaderView::section { background: #edf3ef; border: none; padding: 8px; color: #527263; }
QTabWidget::pane { border: none; background: #ffffff; }
QTabBar::tab { background: #f0f4f1; color: #718078; border-radius: 8px;
    padding: 8px 16px; margin: 0 5px 8px 0; }
QTabBar::tab:selected { background: #dceee3; color: #276149; font-weight: 700; }
QPushButton { background: #ffffff; border: 1px solid #dce5df; border-radius: 9px;
    padding: 8px 13px; color: #3a574a; }
QPushButton:hover { background: #edf5ef; border-color: #99bba8; }
QPushButton:pressed { background: #dceee3; }
QPushButton:disabled { color: #9eada5; background: #f2f5f3; border-color: #e5ebe7; }
QPushButton#primaryButton { background: #367e5e; border-color: #367e5e;
    color: white; font-size: 14px; font-weight: 700; padding: 12px 18px; }
QPushButton#primaryButton:hover { background: #28684b; }
QPushButton#primaryButton:disabled { background: #b7cec0; border-color: #b7cec0; color: #f4f8f5; }
QPushButton#dangerButton { color: #aa6558; }
QPushButton#dangerButton:hover { background: #fbefeb; border-color: #e3bcb0; }
QComboBox { background: #f8faf8; border: 1px solid #dce5df; border-radius: 8px;
    padding: 6px 10px; min-height: 22px; }
QComboBox QAbstractItemView { background: white; selection-background-color: #dceee3; }
QCheckBox { spacing: 8px; padding: 3px 0; }
QProgressBar { background: #e7eee8; border: none; border-radius: 8px;
    min-height: 20px; text-align: center; color: #193f38; }
QProgressBar::chunk { background: #8bc4a2; border-radius: 8px; }
QScrollBar:vertical { background: transparent; width: 8px; margin: 2px; }
QScrollBar::handle:vertical { background: #cbd9cf; border-radius: 3px; min-height: 30px; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0px; }
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { background: transparent; }
QSplitter::handle { background: transparent; width: 14px; }
"""
