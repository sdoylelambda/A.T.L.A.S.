import math
import numpy as np
from queue import Queue
import os

WINDOW_POS_FILE = os.path.join(os.path.dirname(__file__), "..", ".window_pos")
IS_WAYLAND = os.environ.get("WAYLAND_DISPLAY") is not None

from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLineEdit, QLabel, QDesktopWidget
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QObject

from vispy import scene
from scipy.spatial import cKDTree


class FaceSignals(QObject):
    state_changed = pyqtSignal(str)
    caption_changed = pyqtSignal(str)


class FaceController(QMainWindow):
    COLORS = {
        "listening": np.array([0.2, 0.4, 1.0, 1], dtype=np.float32),
        "thinking": np.array([0.2, 1.0, 0.4, 1], dtype=np.float32),
        "error": np.array([1.0, 0.2, 0.2, 1], dtype=np.float32),
        "sleeping": np.array([1.0, 0.9, 0.2, 1], dtype=np.float32),
        "speaking": np.array([0.4, 0.85, 1.0, 1], dtype=np.float32),  # light blue
    }

    PARTICLE_COUNTS = {
        "listening": 400,
        "thinking": 650,
        "error": 400,
        "sleeping": 150,
        "speaking": 550,  # slightly more than listening
    }

    COLOR_VARIATION = {
        "listening": 0.15,
        "thinking": 0.20,
        "error": 0.10,
        "sleeping": 0.08,
        "speaking": 0.12,
    }

    LINE_SETTINGS = {
        "listening": (0.35, 0.25),
        "thinking": (0.4, 0.35),
        "error": (0.3, 0.4),
        "sleeping": (0.5, 0.1),
        "speaking": (0.3, 0.3),  # tighter web, moderate brightness
    }

    BREATH_SETTINGS = {
        "listening": (0.008, 0.03),
        "thinking": (0.015, 0.05),
        "sleeping": (0.004, 0.08),
        "error": (0.025, 0.04),
        "speaking": (0.05, 0.06),  # fast pulse, moderate depth
    }

    BASE_SIZE = 3
    BASE_RADIUS = 1.0

    def __init__(self):
        super().__init__()
        self.setWindowTitle("J.A.R.V.I.S")
        self.setObjectName("jarvis.assistant")
        self.setMinimumSize(220, 350)
        self.setStyleSheet("""
            QMainWindow, QWidget { background-color: #0a0a0f; color: #c8d8e8; }
            QPushButton { background-color: #1a1a2e; color: #c8d8e8; border: 1px solid #2a2a4e;
                          border-radius: 6px; padding: 8px 16px; font-size: 13px; }
            QPushButton:hover { background-color: #2a2a4e; border-color: #4a4a8e; }
            QPushButton:pressed { background-color: #0a0a1e; }
            QPushButton#cancel_btn { border-color: #8e2a2a; color: #ff6b6b; }
            QPushButton#cancel_btn:hover { background-color: #2e1a1a; border-color: #ff4444; }
            QPushButton#mute_btn_active { background-color: #2e1a1a; border-color: #ff4444;
                                          color: #ff6b6b; }
            QLineEdit { background-color: #1a1a2e; color: #c8d8e8; border: 1px solid #2a2a4e;
                        border-radius: 6px; padding: 8px 12px; font-size: 13px; }
            QLineEdit:focus { border-color: #4a6aae; }
            QLabel#caption_label { color: #8a9ab8; font-size: 12px; padding: 4px 8px; min-height: 40px; }
            QLabel#state_label { color: #4a5a7a; font-size: 11px; padding: 2px 8px; }
        """)

        # callbacks — set by Observer after init
        self.on_cancel = None
        self.on_mute = None
        self.on_command = None
        self.muted = False
        self._positioned = False
        self.debug = False

        # signals
        self.signals = FaceSignals()
        self.signals.state_changed.connect(self._apply_state)
        self.signals.caption_changed.connect(self._apply_caption)

        # particle state
        self.current_color = self.COLORS["listening"].copy()
        self.target_color = self.COLORS["listening"].copy()
        self.n_points = 400
        self.points = self._generate_points(self.n_points)
        self.color_offsets = np.random.uniform(-0.1, 0.1, (self.n_points, 4))
        self.color_offsets[:, 3] = 0
        self.base_radius = self.BASE_RADIUS
        self.current_radius = self.base_radius
        self.target_radius = self.base_radius
        self.current_state = "listening"
        self.state_queue = Queue()

        # breathing
        self.breath_phase = 0.0
        self.breath_speed, self.breath_strength = self.BREATH_SETTINGS["listening"]

        # beam lines
        self.beam_counter = 0

        # precompute rotation
        angle = 0.002
        c, s = np.cos(angle), np.sin(angle)
        self.rot_mat = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

        self._build_ui()
        self._start_timer()
        QTimer.singleShot(50, self._restore_position)

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(2)

        # vispy canvas
        self.canvas = scene.SceneCanvas(
            keys='interactive', size=(200, 200),
            show=False, bgcolor='#0a0a0f'
        )
        self.view = self.canvas.central_widget.add_view()
        self.view.camera = scene.cameras.TurntableCamera(
            fov=38, distance=3.5, azimuth=30, elevation=30
        )

        # particles
        self.scatter = scene.visuals.Markers(parent=self.view.scene)
        self.scatter.set_data(self.points, face_color=self.current_color, size=self.BASE_SIZE)

        # beam lines
        self.beam_visual = scene.visuals.Line(
            parent=self.view.scene,
            method='gl',
            connect='segments',
            color=self.current_color
        )

        layout.addWidget(self.canvas.native, stretch=5)

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
        input_layout.setSpacing(2)
        self.text_input = QLineEdit()
        self.text_input.setPlaceholderText("Type a command")
        self.text_input.returnPressed.connect(self._handle_text_command)
        input_layout.addWidget(self.text_input)

        send_btn = QPushButton("Send")
        send_btn.clicked.connect(self._handle_text_command)
        send_btn.setFixedWidth(75)
        input_layout.addWidget(send_btn)
        layout.addLayout(input_layout)

        # buttons
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(2)

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
        self.position_timer.start(5000)

    def _generate_points(self, n):
        theta = np.random.uniform(0, 2 * np.pi, n)
        phi = np.random.uniform(0, np.pi, n)
        r = np.random.uniform(0.2, 1.0, n)
        x = r * np.sin(phi) * np.cos(theta)
        y = r * np.sin(phi) * np.sin(theta)
        z = r * np.cos(phi)
        return np.c_[x, y, z]

    # ── public API ────────────────────────────────────────────────────────

    def set_state(self, state: str):
        if self.debug:
            import traceback
            print(f"[Face] set_state({state})")
            traceback.print_stack(limit=4)
        self.signals.state_changed.emit(state)

    def set_caption(self, text: str):
        self.signals.caption_changed.emit(text)

    # ── Qt slots ──────────────────────────────────────────────────────────

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

    def set_status(self, text: str):
        """Show Brain activity in caption area — thread safe."""
        self.signals.caption_changed.emit(text)

    # ── animation ─────────────────────────────────────────────────────────

    def _update(self):
        # process state changes
        while not self.state_queue.empty():
            state = self.state_queue.get()
            self.target_color = self.COLORS[state].copy()
            self.current_state = state
            self.target_radius = {
                "listening": self.base_radius,
                "thinking": self.base_radius * 1.1,
                "error": self.base_radius * 1.1,
                "sleeping": self.base_radius * 0.5,
                "speaking": self.base_radius * 1.05,  # slightly expanded
            }.get(state, self.base_radius)

            # update breath settings for new state
            self.breath_speed, self.breath_strength = self.BREATH_SETTINGS.get(
                state, (0.008, 0.03)
            )

            # update particle count
            target_count = self.PARTICLE_COUNTS.get(state, 400)
            if target_count != self.n_points:
                self.n_points = target_count
                self.points = self._generate_points(self.n_points)
                self.color_offsets = np.random.uniform(-0.1, 0.1, (self.n_points, 4))
                self.color_offsets[:, 3] = 0

        # smooth color transition
        self.current_color += (self.target_color - self.current_color) * 0.05

        # smooth radius toward target
        self.current_radius += (self.target_radius - self.current_radius) * 0.05

        # breathing — sin wave for perfectly symmetric in/out
        self.breath_phase += self.breath_speed
        if self.breath_phase > 2 * math.pi:
            self.breath_phase -= 2 * math.pi
        breath = math.sin(self.breath_phase)
        display_radius = self.current_radius + breath * self.breath_strength

        # rotate and scale
        self.points = self.points @ self.rot_mat.T
        scaled = self.points * display_radius

        # size variation by distance from center
        dist = np.linalg.norm(scaled, axis=1)
        sizes = self.BASE_SIZE * (1.0 / (0.5 + dist))

        # per-particle color variation
        variation = self.COLOR_VARIATION.get(self.current_state, 0.1)
        per_particle_colors = np.clip(
            self.current_color + self.color_offsets * variation, 0, 1
        )

        self.scatter.set_data(scaled, face_color=per_particle_colors, size=sizes)

        # update beam lines every 5 frames
        self.beam_counter += 1
        if self.beam_counter % 5 == 0:
            result = self._compute_beams(scaled)
            if result is not None:
                verts, colors = result
                self.beam_visual.set_data(
                    pos=verts, color=colors, width=1.2, connect='segments'
                )

        self.canvas.update()

    def _compute_beams(self, points):
        max_dist, max_alpha = self.LINE_SETTINGS.get(self.current_state, (0.4, 0.25))
        subset = points[:200]
        tree = cKDTree(subset)
        pairs = list(tree.query_pairs(max_dist))
        if not pairs:
            return None

        np.random.shuffle(pairs)
        selected = pairs[:len(pairs) // 5]

        verts = []
        colors = []
        base = self.current_color[:3]

        for i, j in selected:
            verts.extend([subset[i], subset[j]])
            dist = np.linalg.norm(subset[i] - subset[j])
            alpha = (1.0 - dist / max_dist) * max_alpha
            colors.extend([[*base, alpha], [*base, alpha]])

        return np.array(verts, dtype=np.float32), np.array(colors, dtype=np.float32)

    # ── window management ─────────────────────────────────────────────────
    # Does not work on Wayland. Should work on just about anything else: x11, mac, windows,etc.

    def showEvent(self, event):
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

    def _save_position(self):
        if IS_WAYLAND:
            return
        try:
            pos = self.pos()
            if pos.x() > 0 or pos.y() > 0:
                with open(WINDOW_POS_FILE, "w") as f:
                    f.write(f"{pos.x()},{pos.y()}")
        except Exception:
            pass

    def closeEvent(self, event):
        self._save_position()
        event.accept()

    def _restore_position(self):
        if IS_WAYLAND:
            return
        try:
            with open(WINDOW_POS_FILE) as f:
                x, y = map(int, f.read().split(","))
                self.move(x, y)
        except Exception:
            pass