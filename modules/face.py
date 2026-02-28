import numpy as np
from queue import Queue
import os
WINDOW_POS_FILE = os.path.join(os.path.dirname(__file__), "..", ".window_pos")

from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLineEdit, QLabel, QSizePolicy, QDesktopWidget
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QObject
from PyQt5.QtGui import QFont, QColor, QPalette

from vispy import scene, app
import vispy.app


class FaceSignals(QObject):
    """Thread-safe signals for Observer → GUI communication."""
    state_changed = pyqtSignal(str)
    caption_changed = pyqtSignal(str)


class FaceController(QMainWindow):
    COLORS = {
        "listening": np.array([0.2, 0.4, 1.0, 1], dtype=np.float32),
        "thinking":  np.array([0.2, 1.0, 0.4, 1], dtype=np.float32),
        "error":     np.array([1.0, 0.2, 0.2, 1], dtype=np.float32),
        "sleeping":  np.array([1.0, 0.9, 0.2, 1], dtype=np.float32),
    }

    BASE_SIZE = 3
    BASE_RADIUS = 1.0

    def __init__(self):
        super().__init__()
        self.setWindowTitle("J.A.R.V.I.S")
        self.setObjectName("jarvis.assistant")
        self.setMinimumSize(400, 550)
        self.setStyleSheet("""
            QMainWindow, QWidget {
                background-color: #0a0a0f;
                color: #c8d8e8;
            }
            QPushButton {
                background-color: #1a1a2e;
                color: #c8d8e8;
                border: 1px solid #2a2a4e;
                border-radius: 6px;
                padding: 8px 16px;
                font-size: 13px;
            }
            QPushButton:hover {
                background-color: #2a2a4e;
                border-color: #4a4a8e;
            }
            QPushButton:pressed {
                background-color: #0a0a1e;
            }
            QPushButton#cancel_btn {
                border-color: #8e2a2a;
                color: #ff6b6b;
            }
            QPushButton#cancel_btn:hover {
                background-color: #2e1a1a;
                border-color: #ff4444;
            }
            QPushButton#mute_btn_active {
                background-color: #2e1a1a;
                border-color: #ff4444;
                color: #ff6b6b;
            }
            QLineEdit {
                background-color: #1a1a2e;
                color: #c8d8e8;
                border: 1px solid #2a2a4e;
                border-radius: 6px;
                padding: 8px 12px;
                font-size: 13px;
            }
            QLineEdit:focus {
                border-color: #4a6aae;
            }
            QLabel#caption_label {
                color: #8a9ab8;
                font-size: 12px;
                padding: 4px 8px;
                min-height: 40px;
            }
            QLabel#state_label {
                color: #4a5a7a;
                font-size: 11px;
                padding: 2px 8px;
            }
        """)

        # callbacks — set by Observer after init
        self.on_cancel = None
        self.on_mute = None
        self.on_command = None
        self.muted = False
        self._positioned = False

        # signals for thread-safe GUI updates
        self.signals = FaceSignals()
        self.signals.state_changed.connect(self._apply_state)
        self.signals.caption_changed.connect(self._apply_caption)

        # particle state
        self.current_color = self.COLORS["listening"].copy()
        self.target_color = self.COLORS["listening"].copy()
        self.n_points = 400
        self.points = self._generate_points(self.n_points)
        self.base_radius = self.BASE_RADIUS
        self.current_radius = self.base_radius
        self.target_radius = self.base_radius
        self.pulse_value = 0.0
        self.pulse_dir = 1
        self.z_wobble = 0.0
        self.z_dir = 1
        self.current_state = "listening"
        self.state_queue = Queue()

        # precompute rotation matrix
        angle = 0.002
        c, s = np.cos(angle), np.sin(angle)
        self.rot_mat = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

        self._build_ui()
        self._start_timer()
        QTimer.singleShot(50, self._restore_position)
        self.debug = False

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        # window size
        self.setMinimumSize(280, 385)
        self.canvas = scene.SceneCanvas(
            keys='interactive', size=(256, 224),
            show=False, bgcolor='#0a0a0f'
        )
        self.view = self.canvas.central_widget.add_view()
        self.view.camera = scene.cameras.TurntableCamera(fov=45, distance=4)
        self.scatter = scene.visuals.Markers(parent=self.view.scene)
        self.scatter.set_data(
            self.points,
            face_color=self.current_color,
            size=self.BASE_SIZE
        )
        layout.addWidget(self.canvas.native, stretch=4)

        # state label
        self.state_label = QLabel("● listening")
        self.state_label.setObjectName("state_label")
        self.state_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.state_label)

        # captions
        self.caption_label = QLabel("")
        self.caption_label.setObjectName("caption_label")
        self.caption_label.setAlignment(Qt.AlignCenter)
        self.caption_label.setWordWrap(True)
        layout.addWidget(self.caption_label)

        # text input
        input_layout = QHBoxLayout()
        self.text_input = QLineEdit()
        self.text_input.setPlaceholderText("Type a command...")
        self.text_input.returnPressed.connect(self._handle_text_command)
        input_layout.addWidget(self.text_input)

        send_btn = QPushButton("Send")
        send_btn.clicked.connect(self._handle_text_command)
        send_btn.setFixedWidth(75)
        input_layout.addWidget(send_btn)
        layout.addLayout(input_layout)

        # cancel and mute buttons
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(8)

        self.cancel_btn = QPushButton("⬛  Cancel")
        self.cancel_btn.setObjectName("cancel_btn")
        self.cancel_btn.clicked.connect(self._handle_cancel)
        btn_layout.addWidget(self.cancel_btn)

        self.mute_btn = QPushButton("🎤  Mute")
        self.mute_btn.setObjectName("mute_btn")
        self.mute_btn.clicked.connect(self._handle_mute)
        btn_layout.addWidget(self.mute_btn)

        layout.addLayout(btn_layout)

    def _start_timer(self):
        self.timer = QTimer()
        self.timer.timeout.connect(self._update)
        self.timer.start(16)  # ~60fps

        self.position_timer = QTimer()
        self.position_timer.timeout.connect(self._save_position)
        self.position_timer.start(5000)  # save position every 5 seconds

    def showEvent(self, event):
        """Called automatically by Qt when window is shown."""
        super().showEvent(event)
        if not self._positioned:
            self._positioned = True
            self.setWindowFlag(Qt.WindowStaysOnTopHint, True)
            self.show()
            screen = QDesktopWidget().availableGeometry()
            self.move(
                screen.right() - self.frameGeometry().width() - 20,
                screen.bottom() - self.frameGeometry().height() - 20
            )

    def _generate_points(self, n):
        theta = np.random.uniform(0, 2 * np.pi, n)
        phi = np.random.uniform(0, np.pi, n)
        r = np.random.uniform(0.5, 1.0, n)
        x = r * np.sin(phi) * np.cos(theta)
        y = r * np.sin(phi) * np.sin(theta)
        z = r * np.cos(phi)
        return np.c_[x, y, z]

    # ── public API (called from Observer thread) ──────────────────────────

    def set_state(self, state: str):
        """Thread-safe state change."""
        self.signals.state_changed.emit(state)

    def set_caption(self, text: str):
        """Thread-safe caption update."""
        self.signals.caption_changed.emit(text)

    # ── Qt slot handlers (always on main thread) ──────────────────────────

    def _apply_state(self, state: str):
        if state not in self.COLORS:
            return
        self.state_queue.put(state)
        labels = {
            "listening": "● listening",
            "thinking":  "● thinking",
            "error":     "● error",
            "sleeping":  "● sleeping",
        }
        colors = {
            "listening": "#3a6aee",
            "thinking":  "#3aee6a",
            "error":     "#ee3a3a",
            "sleeping":  "#eec83a",
        }
        self.state_label.setText(labels.get(state, ""))
        self.state_label.setStyleSheet(
            f"color: {colors.get(state, '#4a5a7a')}; font-size: 11px; padding: 2px 8px;"
        )

    def _apply_caption(self, text: str):
        self.caption_label.setText(text)

    def _handle_cancel(self):
        if self.on_cancel:
            self.on_cancel()
        self.set_state("listening")
        self.set_caption("")

    def _handle_mute(self):
        self.muted = not self.muted
        if self.muted:
            self.mute_btn.setObjectName("mute_btn_active")
            self.mute_btn.setText("🔇  Unmute")
            self.mute_btn.setStyleSheet(
                "background-color: #2e1a1a; border: 1px solid #ff4444; "
                "color: #ff6b6b; border-radius: 6px; padding: 8px 16px; font-size: 13px;"
            )
        else:
            self.mute_btn.setObjectName("mute_btn")
            self.mute_btn.setText("🎤  Mute")
            self.mute_btn.setStyleSheet("")
        if self.on_mute:
            self.on_mute(self.muted)

    def _handle_text_command(self):
        text = self.text_input.text().strip()
        if text and self.on_command:
            self.text_input.clear()
            self.set_state("thinking")
            self.on_command(text)

    # ── particle animation (called by QTimer on main thread) ──────────────

    def _update(self):
        while not self.state_queue.empty():
            state = self.state_queue.get()
            self.target_color = self.COLORS[state].copy()
            self.current_state = state
            self.target_radius = {
                "listening": self.base_radius,
                "thinking":  self.base_radius * 1.1,
                "error":     self.base_radius * 1.1,
                "sleeping":  self.base_radius * 0.5,
            }.get(state, self.base_radius)
            self.pulse_value = 0.0
            self.pulse_dir = 1
            self.z_wobble = 0.0
            self.z_dir = 1

        self.current_color += (self.target_color - self.current_color) * 0.05

        params = {
            "thinking":  (0.001,      self.target_radius * 0.03, 0.001,   0.005),
            "listening": (0.0000005,  self.target_radius * 0.02, 0.00005, 0.0003),
            "error":     (0.000005,   self.target_radius * 0.03, 0.005,   0.05),
            "sleeping":  (0.0005,     self.target_radius * 0.05, 0.0003,  0.002),
        }.get(self.current_state, (0, 0, 0, 0))

        pulse_speed, pulse_strength, z_speed, z_strength = params

        self.pulse_value += self.pulse_dir * pulse_speed
        if abs(self.pulse_value) > pulse_strength:
            self.pulse_dir *= -1

        self.current_radius += (
            self.target_radius + self.pulse_value - self.current_radius
        ) * 0.3

        self.z_wobble += self.z_dir * z_speed
        if abs(self.z_wobble) > z_strength:
            self.z_dir *= -1

        self.points = self.points @ self.rot_mat.T
        scaled = self.points * self.current_radius
        scaled[:, 2] += self.z_wobble

        self.scatter.set_data(scaled, face_color=self.current_color, size=self.BASE_SIZE)
        self.canvas.update()

    # on close, save face window location to a small file to have it load in the same place
    def _save_position(self):
        try:
            pos = self.pos()
            if pos.x() > 0 or pos.y() > 0:
                with open(WINDOW_POS_FILE, "w") as f:
                    f.write(f"{pos.x()},{pos.y()}")
                if self.debug:
                    print(f"[GUI] Position saved: {pos.x()},{pos.y()} to {WINDOW_POS_FILE}")
        except Exception as e:
            print(f"[GUI] Position save failed: {e}")

    def closeEvent(self, event):
        pos = self.pos()
        with open(".window_pos", "w") as f:
            f.write(f"{pos.x()},{pos.y()}")
        event.accept()

    # on start, restore if file exists - remembers last place window was located on screen
    def _restore_position(self):
        try:
            with open(WINDOW_POS_FILE) as f:
                x, y = map(int, f.read().split(","))
                self.move(x, y)
                if self.debug:
                    print(f"[GUI] Position restored: {x},{y}")
        except Exception as e:
            print(f"[GUI] Position restore failed: {e}")
