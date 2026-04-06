from __future__ import annotations

from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QApplication
from PySide6.QtCore import Qt, QTimer, QPointF, QRectF, Property
from PySide6.QtGui import QPainter, QColor, QConicalGradient, QPalette, QFont
import math


class OptimizingSpinner(QWidget):
    """
    Високооптимізований, естетичний віджет очікування для довгих обчислень.
    Використовує анімацію 'дихання' та обертання градієнта для заспокійливого ефекту.
    """

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)

        # Акцентний колір (беремо з Qt Palette або за замовчуванням)
        self._base_color = QApplication.instance().palette().color(QPalette.Highlight)
        if not self._base_color.isValid():
            self._base_color = QColor("#2979FF")  # Фоллбек на твій синій

        self._angle = 0
        self._opacity = 1.0
        self._animation_step = 0

        # 1. Основний таймер анімації обертання (дуже швидкий, але легкий)
        self._rotation_timer = QTimer(self)
        self._rotation_timer.timeout.connect(self._rotate)
        self._rotation_timer.setInterval(16)  # ~60 FPS для гладкості

        # 2. Таймер для ефекту 'дихання' (плавне мерехтіння)
        self._pulse_timer = QTimer(self)
        self._pulse_timer.timeout.connect(self._pulse)
        self._pulse_timer.setInterval(50)  # Повільніше

        self._setup_ui()
        self.hide()  # Прихований за замовчуванням

    def _setup_ui(self):
        """Налаштування лейауту та текстового повідомлення."""
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignCenter)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(15)

        # Простір для самого спінера (він малюється у paintEvent)
        self._spinner_spacer = QWidget()
        self._spinner_spacer.setFixedSize(60, 60)
        layout.addWidget(self._spinner_spacer, 0, Qt.AlignCenter)

        # Текстовий лейбл
        self.label = QLabel("Оптимізація портфеля...")
        font = QFont()
        font.setBold(True)
        font.setPointSize(11)
        self.label.setFont(font)
        self.label.setAlignment(Qt.AlignCenter)

        self.detail_label = QLabel("")
        self.detail_label.setAlignment(Qt.AlignCenter)
        detail_font = QFont()
        detail_font.setPointSize(10)
        self.detail_label.setFont(detail_font)
        self.detail_label.setStyleSheet("color: rgba(128, 128, 128, 0.8);")  # Тьмяніший колір
        layout.addWidget(self.detail_label)

        # Використовуємо адаптивний колір тексту з палітри
        palette = QApplication.instance().palette()
        text_color = palette.color(QPalette.WindowText)
        # Трохи приглушуємо колір тексту для заспокійливого ефекту
        text_color.setAlpha(180)
        self.label.setStyleSheet(
            f"color: rgba({text_color.red()}, {text_color.green()}, {text_color.blue()}, {text_color.alpha() / 255.0});"
        )

        layout.addWidget(self.label)

    def update_progress(self, percent: int, detail_text: str = ""):
        """Оновлює текст прогресу без переривання анімації."""
        if detail_text:
            self.detail_label.setText(f"{detail_text} ({percent}%)")
        else:
            self.detail_label.setText(f"{percent}%")

    def start(self, message: str | None = None):
        """Запускає анімацію та показує віджет."""
        if message:
            self.label.setText(message)
        self.show()
        self._rotation_timer.start()
        self._pulse_timer.start()
        self._animation_step = 0

    def stop(self):
        """Зупиняє анімацію та ховає віджет."""
        self._rotation_timer.stop()
        self._pulse_timer.stop()
        self.hide()

    def _rotate(self):
        """Оновлює кут повороту."""
        self._angle = (self._angle + 4) % 360
        self.update()  # Викликає paintEvent

    def _pulse(self):
        """Створює ефект повільного дихання (зміна прозорості)."""
        self._animation_step += 1
        # Використовуємо синусоїду для плавного коливання прозорості від 0.6 до 1.0
        self._opacity = 0.8 + 0.2 * math.sin(self._animation_step * 0.1)
        # update() не потрібен, rotation_timer вже викликає його часто

    def paintEvent(self, event):
        """Малює кастомний, красивий спінер."""
        if not self.isVisible():
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)  # Максимальна гладкість
        painter.setRenderHint(QPainter.SmoothPixmapTransform)

        # Отримуємо геометрію верхнього спайсера
        geom = self._spinner_spacer.geometry()
        center = QPointF(geom.center().x(), geom.center().y())
        radius = min(geom.width(), geom.height()) / 2

        # Налаштування пензля
        pen_width = 4.0
        draw_rect = QRectF(center.x() - radius + pen_width, center.y() - radius + pen_width,
                           (radius - pen_width) * 2, (radius - pen_width) * 2)

        # 1. Малюємо напівпрозоре базове коло (фон спінера)
        base_pen_color = QColor(self._base_color)
        base_pen_color.setAlpha(30)  # Дуже слабкий фон
        painter.setPen(Qt.NoPen)
        painter.setBrush(base_pen_color)
        painter.drawEllipse(center, radius - pen_width / 2, radius - pen_width / 2)

        # 2. Малюємо обертовий градієнт
        painter.save()
        painter.translate(center)
        painter.rotate(self._angle)  # Обертаємо всю систему координат

        # Створюємо конічний градієнт (хвіст спінера)
        gradient = QConicalGradient(QPointF(0, 0), 0)

        # Застосовуємо ефект 'дихання' (прозорість) до базового кольору
        main_color = QColor(self._base_color)
        main_color.setAlphaF(self._opacity)

        # Градієнт від основного кольору до повної прозорості
        gradient.setColorAt(0.0, main_color)
        gradient.setColorAt(0.5, QColor(0, 0, 0, 0))  # Прозорий хвіст
        gradient.setColorAt(1.0, main_color)

        # Використовуємо градієнт як пензель для дуги
        from PySide6.QtGui import QPen, QBrush
        grad_pen = QPen(QBrush(gradient), pen_width, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
        painter.setPen(grad_pen)

        # Малюємо дугу (не повне коло, щоб було видно обертання)
        # Координати відносні, бо ми зробили painter.translate(center)
        arc_rect = QRectF(-radius + pen_width, -radius + pen_width,
                          (radius - pen_width) * 2, (radius - pen_width) * 2)

        # Малюємо дугу довжиною 270 градусів
        painter.drawArc(arc_rect, 0 * 16, 270 * 16)

        painter.restore()


# ─── Тестування (можна запустити файл напряму) ───
if __name__ == "__main__":
    import sys

    app = QApplication(sys.argv)

    # Світла тема для тесту
    # app.setStyle("Fusion")

    # Темна тема для тесту
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor("#1A1A1A"))
    palette.setColor(QPalette.WindowText, Qt.white)
    palette.setColor(QPalette.Highlight, QColor("#2979FF"))
    app.setPalette(palette)

    window = QWidget()
    window.setWindowTitle("Spinner Test")
    window.setFixedSize(400, 300)
    layout = QVBoxLayout(window)

    spinner = OptimizingSpinner(window)
    layout.addWidget(spinner)

    spinner.start("Завантаження даних...")  # Запуск

    # Через 5 секунд міняємо текст (імітація зміни етапу)
    QTimer.singleShot(5000, lambda: spinner.label.setText("Запуск бектесту (це займе час)..."))

    window.show()
    sys.exit(app.exec())