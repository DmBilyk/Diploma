import re

with open('/Users/macbook/Documents/Diploma/app/ui/widget/backtest_widget.py', 'r') as f:
    content = f.read()

# 1. Remove _BG and _TEXT definitions
content = re.sub(r'_BG0\s*=\s*"[^"]+"\n_BG1.*?\n_TEXT2\s*=\s*"[^"]+"\n', '', content, flags=re.DOTALL)

# Default modern accent
content = re.sub(r'_ACCENT\s*=\s*"[^"]+"', '_ACCENT  = "#007AFF"', content)

# 2. Remove _input_style() entirely
content = re.sub(r'# ── Загальний стиль.*?(?=# ══════════════════════════════════════════════════════════════════════════════\n#  MATPLOTLIB КАНВАС)', '', content, flags=re.DOTALL)

# 3. Rename _DarkCanvas to _ThemeAwareCanvas and update its implementation
canvas_old = """class _DarkCanvas(FigureCanvas):
    def __init__(self, fig: Figure, parent: QWidget | None = None) -> None:
        super().__init__(fig)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.figure.patch.set_facecolor(_BG0)"""
canvas_new = """class _ThemeAwareCanvas(FigureCanvas):
    def __init__(self, fig: Figure, parent: QWidget | None = None) -> None:
        super().__init__(fig)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.figure.patch.set_alpha(0.0)
        self.setStyleSheet("background-color: transparent;")"""
content = content.replace(canvas_old, canvas_new)
content = content.replace('_DarkCanvas', '_ThemeAwareCanvas')

# 4. _ControlPanel styles
content = re.sub(r'self\.setStyleSheet\(f"""\s*QScrollArea \{.*?\}"""\)', '', content, flags=re.DOTALL)
content = re.sub(r'self\._inner\.setStyleSheet\(f"background: \{_BG1\};"\)', '', content)

# algo_box and bench_box
content = re.sub(r'algo_box\.setStyleSheet\(f"""\s*QGroupBox \{.*?\}\s*"""\)', '', content, flags=re.DOTALL)
content = re.sub(r'bench_box\.setStyleSheet\(algo_box\.styleSheet\(\)\)', '', content)

# sep line
content = content.replace('sep.setStyleSheet(f"color: {_BG3}; margin-top: 4px; margin-bottom: 4px;")', 'sep.setStyleSheet("color: palette(mid); margin-top: 4px; margin-bottom: 4px;")')

# combo plugin
content = content.replace('self._combo_plugin.setStyleSheet(_input_style() + " QComboBox { padding: 2px 6px; }")', '')

# btn run
btn_run_old = """        self.btn_run.setStyleSheet(f\"\"\"
            QPushButton {{
                background-color: {_ACCENT};
                color: #0d1117;
                border: none;
                border-radius: 6px;
                font-size: 13px;
                font-weight: 800;
                letter-spacing: 0.5px;
            }}
            QPushButton:hover   {{ background-color: #16a085; }}
            QPushButton:pressed {{ background-color: #0e7a62; }}
            QPushButton:disabled {{
                background-color: {_BG3};
                color: {_TEXT2};
            }}
        \"\"\")"""
btn_run_new = """        self.btn_run.setStyleSheet(f\"\"\"
            QPushButton {{
                background-color: {_ACCENT};
                color: white;
                border: none;
                border-radius: 6px;
                font-size: 13px;
                font-weight: bold;
                padding: 10px;
            }}
            QPushButton:hover   {{ background-color: #006ae0; }}
            QPushButton:pressed {{ background-color: #0056b3; }}
            QPushButton:disabled {{
                background-color: #cccccc;
                color: #666666;
            }}
        \"\"\")"""
content = content.replace(btn_run_old, btn_run_new)

# field labels
content = re.sub(r'lbl\.setStyleSheet\(.*? color: \{_ACCENT\}.*?\)', '', content)
content = content.replace('lbl.setStyleSheet(f"color: {_TEXT2}; font-size: 10px;")', '')
content = re.sub(r'self\._section_label\("([^"]+)"\)', r'QLabel("<b>\1</b>")', content)
content = re.sub(r'def _section_label[^)]+\) -> QLabel:\n.*?return lbl\n', '', content, flags=re.DOTALL)
content = re.sub(r'def _field_label[^)]+\) -> QLabel:\n.*?return lbl\n', '', content, flags=re.DOTALL)
content = re.sub(r'self\._field_label\("([^"]+)"\)', r'QLabel("\1")', content)

lbl_section_old = """    @staticmethod
    def _section_label(text: str) -> QLabel:
        lbl = QLabel(text.upper())
        lbl.setStyleSheet(
            f"color: {_ACCENT}; font-size: 10px; font-weight: 700;"
            f" letter-spacing: 1.5px; padding-top: 2px;"
        )
        return lbl

    @staticmethod
    def _field_label(text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setStyleSheet(f"color: {_TEXT2}; font-size: 10px;")
        return lbl"""
content = content.replace(lbl_section_old, "")

# make_checkbox
cb_old = """    @staticmethod
    def _make_checkbox(text: str, color: str, checked: bool = True) -> QCheckBox:
        cb = QCheckBox(text)
        cb.setChecked(checked)
        cb.setStyleSheet(f\"\"\"
            QCheckBox {{
                color: {_TEXT1};
                font-size: 12px;
                spacing: 8px;
            }}
            QCheckBox::indicator {{
                width: 16px;
                height: 16px;
                border: 2px solid {_BG3};
                border-radius: 4px;
                background: {_BG2};
            }}
            QCheckBox::indicator:checked {{
                background: {color};
                border-color: {color};
            }}
            QCheckBox::indicator:hover {{
                border-color: {color};
            }}
        \"\"\")
        return cb"""
cb_new = """    @staticmethod
    def _make_checkbox(text: str, color: str, checked: bool = True) -> QCheckBox:
        cb = QCheckBox(text)
        cb.setChecked(checked)
        # Using a left border or simply coloring the text to indicate series color
        cb.setStyleSheet(f"QCheckBox {{ color: palette(windowText); }} QCheckBox::indicator:checked {{ background-color: {color}; }}")
        return cb"""
content = content.replace(cb_old, cb_new)

# make_date, make_spin, make_dspin
content = content.replace('w.setStyleSheet(_input_style())\n        # Застосовуємо темний стиль до календаря після відкриття\n        w.calendarWidget().setStyleSheet(_input_style())', '')
content = content.replace('w.setStyleSheet(_input_style())', '')

# _MetricsTable
table_old = """        self.setStyleSheet(f\"\"\"
            QTableWidget {{
                background-color: {_BG2};
                color: {_TEXT0};
                border: 1px solid {_BG3};
                border-radius: 8px;
                gridline-color: {_BG3};
                font-size: 11px;
            }}
            QHeaderView::section {{
                background-color: {_BG1};
                color: {_TEXT2};
                border: none;
                border-bottom: 1px solid {_BG3};
                padding: 6px 8px;
                font-size: 10px;
                font-weight: 700;
                letter-spacing: 0.5px;
            }}
            QTableWidget::item {{
                padding: 5px 8px;
                border: none;
            }}
            QTableWidget::item:selected {{
                background-color: {_BG3};
                color: {_TEXT0};
            }}
        \"\"\")"""
content = content.replace(table_old, "")

# _ComparisonDashboard
content = content.replace('self.setStyleSheet(f"background: {_BG1};")', '')
content = content.replace('self._weights_selector.setStyleSheet(_input_style())', '')
content = content.replace('color=_TEXT1', 'color="palette(windowText)"') # wait matplotlib requires rgb or hex, we need theme aware text

# We will rewrite the drawing logic later to be theme aware.
# Let's fix Matplotlib theme aware text
# Matplotlib text colors: instead of _TEXT1, we can pass text_color from palette
content = content.replace('_TEXT1', 'text_color')
content = content.replace('_TEXT2', 'grid_color')
content = content.replace('_BG3', 'grid_color')
content = content.replace('_BG2', 'bg_color')
content = content.replace('_BG0', 'bg_color')

# Need to update _draw_* methods to extract colors
draw_equity_old = """    def _draw_equity(self) -> None:
        fig = self._fig_main
        fig.clear()
        ax = fig.add_subplot(111)
        self._style_ax(ax)"""
draw_equity_new = """    def _get_theme_colors(self):
        pal = self.palette()
        is_dark = pal.color(pal.Window).lightness() < 128
        text_color = "#e6edf3" if is_dark else "#24292f"
        grid_color = "#30363d" if is_dark else "#d0d7de"
        bg_color   = "#21262d" if is_dark else "#ffffff"
        return text_color, grid_color, bg_color

    def _draw_equity(self) -> None:
        text_color, grid_color, bg_color = self._get_theme_colors()
        fig = self._fig_main
        fig.clear()
        ax = fig.add_subplot(111)
        self._style_ax(ax, text_color, grid_color, bg_color)"""
content = content.replace(draw_equity_old, draw_equity_new)

draw_dd_old = """    def _draw_drawdown(self) -> None:
        fig = self._fig_dd
        fig.clear()
        ax = fig.add_subplot(111)
        self._style_ax(ax)"""
draw_dd_new = """    def _draw_drawdown(self) -> None:
        text_color, grid_color, bg_color = self._get_theme_colors()
        fig = self._fig_dd
        fig.clear()
        ax = fig.add_subplot(111)
        self._style_ax(ax, text_color, grid_color, bg_color)"""
content = content.replace(draw_dd_old, draw_dd_new)

draw_w_old = """    def _draw_weights(self, series: dict | None) -> None:
        fig = self._fig_w
        fig.clear()
        ax = fig.add_subplot(111)
        self._style_ax(ax)"""
draw_w_new = """    def _draw_weights(self, series: dict | None) -> None:
        text_color, grid_color, bg_color = self._get_theme_colors()
        fig = self._fig_w
        fig.clear()
        ax = fig.add_subplot(111)
        self._style_ax(ax, text_color, grid_color, bg_color)"""
content = content.replace(draw_w_old, draw_w_new)

style_ax_old = """    @staticmethod
    def _style_ax(ax) -> None:
        ax.tick_params(colors=grid_color, which="both", labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor(grid_color)
        ax.set_facecolor(bg_color)"""
style_ax_new = """    @staticmethod
    def _style_ax(ax, text_color, grid_color, bg_color) -> None:
        ax.tick_params(colors=text_color, which="both", labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor(grid_color)
        ax.set_facecolor("none")"""
content = content.replace(style_ax_old, style_ax_new)

# _EmptyState
content = content.replace('self.setStyleSheet(f"background: {_BG1};")', '')
content = re.sub(r'title\.setStyleSheet\(\n\s*f"color.*?"\n\s*\)', 'title.setStyleSheet("font-size: 20px; font-weight: 700;")', content)
content = re.sub(r'subtitle\.setStyleSheet\(\n\s*f"color.*?"\n\s*\)', 'subtitle.setStyleSheet("font-size: 13px; padding-top: 10px;")', content)
content = re.sub(r'hint\.setStyleSheet\(\n\s*f"color.*?"\n\s*\)', 'hint.setStyleSheet("font-size: 10px; padding-top: 6px;")', content)

# _SpinnerView
content = content.replace('self.setStyleSheet(f"background: {_BG1};")', '')

# BacktestWidget
content = content.replace('self.setStyleSheet(f"background: {_BG1};")', '')
content = content.replace('self._result_stack.setStyleSheet(f"background: {_BG1};")', '')
content = content.replace('bar.setStyleSheet(f"background: {_BG1};")', '')
content = content.replace('title.setStyleSheet(\n            f"color: {_TEXT0}; font-size: 18px; font-weight: 700; letter-spacing: 0.5px;"\n        )', 'title.setStyleSheet("font-size: 18px; font-weight: 700; letter-spacing: 0.5px;")')
content = re.sub(r'self\._status_badge\.setStyleSheet\(f"""\s*background: \{_BG2\}; color: \{_TEXT2\};\s*border: 1px solid \{_BG3\}; border-radius: 10px;\s*padding: 3px 12px; font-size: 11px; font-weight: 600;\s*"""\)', 'self._status_badge.setStyleSheet("border-radius: 10px; padding: 3px 12px; font-size: 11px; font-weight: 600; border: 1px solid palette(mid);")', content)

content = content.replace('f.setStyleSheet(f"background: {_BG3}; border: none;")', 'f.setStyleSheet("background: palette(mid); border: none;")')

set_status_old = """        self._status_badge.setStyleSheet(f\"\"\"
            background: {bg_color}; color: {color};
            border: 1px solid {color}; border-radius: 10px;
            padding: 3px 12px; font-size: 11px; font-weight: 600;
        \"\"\")"""
set_status_new = """        self._status_badge.setStyleSheet(f\"\"\"
            color: {color};
            border: 1px solid {color}; border-radius: 10px;
            padding: 3px 12px; font-size: 11px; font-weight: 600;
        \"\"\")"""
content = content.replace(set_status_old, set_status_new)

with open('/Users/macbook/Documents/Diploma/app/ui/widget/backtest_widget.py', 'w') as f:
    f.write(content)
