from vispy import scene, app
import numpy as np
from queue import Queue

class FaceController:
    COLORS = {
        "listening": np.array([0, 0, 1, 1], dtype=np.float32),
        "thinking": np.array([0, 1, 0, 1], dtype=np.float32),
        "error": np.array([1, 0, 0, 1], dtype=np.float32),
        "sleeping": np.array([1, 1, 0, 1], dtype=np.float32),  # yellow
    }

    BASE_SIZE = 3
    BASE_RADIUS = 1.0

    def __init__(self):
        self.state_queue = Queue()
        self.current_color = self.COLORS["thinking"].copy()
        self.target_color = self.COLORS["thinking"].copy()
        self.n_points = 400
        self.points = self.generate_points(self.n_points)

        # Pulse / breathing
        self.base_radius = self.BASE_RADIUS
        self.current_radius = self.base_radius
        self.target_radius = self.base_radius
        self.pulse_value = 0.0
        self.pulse_dir = 1
        self.z_wobble = 0.0
        self.z_dir = 1
        self.current_state = "listening"

        # SceneCanvas
        self.canvas = scene.SceneCanvas(keys='interactive', size=(250, 250),
                                        show=True, title="Jarvis's AI Face")
        self.view = self.canvas.central_widget.add_view()
        self.view.camera = scene.cameras.TurntableCamera(fov=45, distance=4)
        self.scatter = scene.visuals.Markers(parent=self.view.scene)
        self.scatter.set_data(self.points, face_color=self.current_color, size=self.BASE_SIZE)

        # Precompute slow rotation matrix
        self.angle = 0.002
        c, s = np.cos(self.angle), np.sin(self.angle)
        self.rot_mat = np.array([
            [c, -s, 0],
            [s,  c, 0],
            [0,  0, 1]
        ])

        # Timer for updates
        self.timer = app.Timer(interval=1/60.0, connect=self.update, start=True)

    def generate_points(self, n):
        theta = np.random.uniform(0, 2*np.pi, n)
        phi = np.random.uniform(0, np.pi, n)
        r = np.random.uniform(0.5, 1.0, n)
        x = r * np.sin(phi) * np.cos(theta)
        y = r * np.sin(phi) * np.sin(theta)
        z = r * np.cos(phi)
        return np.c_[x, y, z]

    def set_state(self, state: str):
        if state in self.COLORS:
            self.state_queue.put(state)

    def update(self, event):
        # Apply queued state changes smoothly
        while not self.state_queue.empty():
            state = self.state_queue.get()
            self.target_color = self.COLORS[state].copy()
            self.current_state = state

            if state == "listening":
                self.target_radius = self.base_radius
            elif state == "thinking":
                self.target_radius = self.base_radius * 1.1
            elif state == "sleeping":
                self.target_radius = self.base_radius * 0.5  # exactly half-size
            else:  # error or default
                self.target_radius = self.base_radius

            # Reset pulse/wobble
            self.pulse_value = 0.0
            self.pulse_dir = 1
            self.z_wobble = 0.0
            self.z_dir = 1

        # Smooth color transition
        self.current_color += (self.target_color - self.current_color) * 0.05

        # Pulse & wobble per state
        if self.current_state == "thinking":
            pulse_speed = 0.001
            pulse_strength = self.target_radius * 0.03
            z_speed = 0.001
            z_strength = 0.005
        elif self.current_state == "listening":
            pulse_speed = 0.0000005
            pulse_strength = self.target_radius * 0.02
            z_speed = 0.00005
            z_strength = 0.0003
        elif self.current_state == "error":
            pulse_speed = 0.000005
            pulse_strength = self.target_radius * 0.03
            z_speed = 0.005
            z_strength = 0.05
        elif self.current_state == "sleeping":
            pulse_speed = 0.0005  # very slow
            pulse_strength = self.target_radius * 0.05  # tiny pulse relative to half-size
            z_speed = 0.0003  # gentle wobble
            z_strength = 0.002
        else:  # error or default
            pulse_speed = 0
            pulse_strength = 0
            z_speed = 0
            z_strength = 0

        # Update pulse (grow/shrink)
        self.pulse_value += self.pulse_dir * pulse_speed
        if self.pulse_value > pulse_strength or self.pulse_value < -pulse_strength:
            self.pulse_dir *= -1

        # Smooth radius toward target + pulse
        self.current_radius += (self.target_radius + self.pulse_value - self.current_radius) * 0.3

        # Update z-axis wobble
        self.z_wobble += self.z_dir * z_speed
        if self.z_wobble > z_strength or self.z_wobble < -z_strength:
            self.z_dir *= -1

        # Rotate points slowly
        self.points = self.points @ self.rot_mat.T

        # Apply radius and z wobble
        scaled_points = self.points * self.current_radius
        scaled_points[:, 2] += self.z_wobble

        # Update scatter
        self.scatter.set_data(scaled_points, face_color=self.current_color, size=self.BASE_SIZE)

    def run(self):
        app.run()
