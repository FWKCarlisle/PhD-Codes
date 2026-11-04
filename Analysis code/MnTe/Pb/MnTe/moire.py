import sys
import numpy as np

from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QSlider,
    QDoubleSpinBox,
    QSpinBox,
    QGroupBox,
)
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


def create_hexagonal_lattice(rows, cols, a):
    points = []
    a1 = np.array([a, 0.0])
    a2 = np.array([a / 2.0, a * np.sqrt(3) / 2.0])

    for i in range(rows):
        for j in range(cols):
            p = i * a1 + j * a2
            points.append(p)

    return np.array(points)


def rotate_points(points, theta):
    c = np.cos(theta)
    s = np.sin(theta)
    R = np.array([[c, -s],
                  [s,  c]])
    return points @ R.T


def moire_lattice_constant(a1, a2, theta_deg):
    """
    Moire wavelength for two lattices with different lattice constants
    and a relative twist angle theta_deg.

    For identical lattices, a1 = a2 = a, this reduces to:
        Lm = a / (2 * sin(theta/2))
    """
    theta = np.deg2rad(theta_deg)
    if np.isclose(theta, 0.0):
        return np.inf

    denom = np.sqrt(
        (1 / a1**2) + (1 / a2**2) - 2 * np.cos(theta) / (a1 * a2)
    )
    return 1.0 / denom


class MoireCanvas(FigureCanvas):
    def __init__(self, parent=None):
        self.fig = Figure(figsize=(7, 7))
        self.ax = self.fig.add_subplot(111)
        super().__init__(self.fig)

        if parent is not None:
            self.setParent(parent)

        self.ax.set_aspect("equal")
        self.ax.set_xlabel("x (Å)")
        self.ax.set_ylabel("y (Å)")
        self.fig.tight_layout()

    def plot_lattices(self, a1, a2, theta_deg, rows1=24, cols1=24, rows2=24, cols2=24):
        self.ax.clear()

        p1 = create_hexagonal_lattice(rows1, cols1, a1)
        p1 = p1 - p1.mean(axis=0)

        p2 = create_hexagonal_lattice(rows2, cols2, a2)
        p2 = rotate_points(p2, np.deg2rad(theta_deg))
        p2 = p2 - p2.mean(axis=0)

        self.ax.scatter(
            p1[:, 0], p1[:, 1],
            s=16, color="tab:blue", alpha=0.8,
            label=f"Lattice 1 (a={a1:.3f} Å, {rows1}x{cols1})"
        )

        self.ax.scatter(
            p2[:, 0], p2[:, 1],
            s=16, color="tab:orange", alpha=0.8,
            label=f"Lattice 2 (a={a2:.3f} Å, {rows2}x{cols2})"
        )

        self.ax.set_title(
            f"Moiré lattice, θ = {theta_deg:.2f}°, "
            f"a1={a1:.3f} Å, a2={a2:.3f} Å"
        )
        self.ax.legend()
        self.ax.set_aspect("equal")
        self.draw_idle()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Moiré Lattice GUI")
        self.resize(1100, 700)

        self.a1 = 4.92
        self.a2 = 4.92
        self.theta_deg = 10.0

        self.rows1 = 24
        self.cols1 = 24
        self.rows2 = 24
        self.cols2 = 24

        central = QWidget(self)
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)

        controls = QGroupBox("Controls")
        controls_layout = QHBoxLayout()

        # Lattice 1 controls
        lattice1_layout = QVBoxLayout()
        a1_label = QLabel("Lattice 1: a1 (Å)")
        self.a1_spin = QDoubleSpinBox()
        self.a1_spin.setRange(0.5, 20.0)
        self.a1_spin.setSingleStep(0.01)
        self.a1_spin.setDecimals(3)
        self.a1_spin.setValue(self.a1)
        lattice1_layout.addWidget(a1_label)
        lattice1_layout.addWidget(self.a1_spin)

        rows1_label = QLabel("Rows 1")
        self.rows1_spin = QSpinBox()
        self.rows1_spin.setRange(2, 200)
        self.rows1_spin.setValue(self.rows1)
        lattice1_layout.addWidget(rows1_label)
        lattice1_layout.addWidget(self.rows1_spin)

        cols1_label = QLabel("Cols 1")
        self.cols1_spin = QSpinBox()
        self.cols1_spin.setRange(2, 200)
        self.cols1_spin.setValue(self.cols1)
        lattice1_layout.addWidget(cols1_label)
        lattice1_layout.addWidget(self.cols1_spin)

        controls_layout.addLayout(lattice1_layout)

        # Lattice 2 controls
        lattice2_layout = QVBoxLayout()
        a2_label = QLabel("Lattice 2: a2 (Å)")
        self.a2_spin = QDoubleSpinBox()
        self.a2_spin.setRange(0.5, 20.0)
        self.a2_spin.setSingleStep(0.01)
        self.a2_spin.setDecimals(3)
        self.a2_spin.setValue(self.a2)
        lattice2_layout.addWidget(a2_label)
        lattice2_layout.addWidget(self.a2_spin)

        rows2_label = QLabel("Rows 2")
        self.rows2_spin = QSpinBox()
        self.rows2_spin.setRange(2, 200)
        self.rows2_spin.setValue(self.rows2)
        lattice2_layout.addWidget(rows2_label)
        lattice2_layout.addWidget(self.rows2_spin)

        cols2_label = QLabel("Cols 2")
        self.cols2_spin = QSpinBox()
        self.cols2_spin.setRange(2, 200)
        self.cols2_spin.setValue(self.cols2)
        lattice2_layout.addWidget(cols2_label)
        lattice2_layout.addWidget(self.cols2_spin)

        controls_layout.addLayout(lattice2_layout)

        # Angle controls
        angle_layout = QVBoxLayout()
        angle_label = QLabel("Twist angle θ (deg)")
        self.angle_spin = QDoubleSpinBox()
        self.angle_spin.setRange(0.1, 60.0)
        self.angle_spin.setSingleStep(0.1)
        self.angle_spin.setDecimals(2)
        self.angle_spin.setValue(self.theta_deg)
        angle_layout.addWidget(angle_label)
        angle_layout.addWidget(self.angle_spin)

        slider_label = QLabel("Angle slider")
        self.angle_slider = QSlider(Qt.Horizontal)
        self.angle_slider.setRange(1, 600)
        self.angle_slider.setValue(int(self.theta_deg * 10))
        angle_layout.addWidget(slider_label)
        angle_layout.addWidget(self.angle_slider)

        controls_layout.addLayout(angle_layout)

        controls.setLayout(controls_layout)
        main_layout.addWidget(controls)

        # Result label
        self.result_label = QLabel()
        self.result_label.setStyleSheet("font-size: 14px; font-weight: bold;")
        main_layout.addWidget(self.result_label)

        # Plot
        self.canvas = MoireCanvas(self)
        main_layout.addWidget(self.canvas)

        # Connect signals
        self.a1_spin.valueChanged.connect(self.update_parameters)
        self.a2_spin.valueChanged.connect(self.update_parameters)
        self.rows1_spin.valueChanged.connect(self.update_parameters)
        self.cols1_spin.valueChanged.connect(self.update_parameters)
        self.rows2_spin.valueChanged.connect(self.update_parameters)
        self.cols2_spin.valueChanged.connect(self.update_parameters)

        self.angle_spin.valueChanged.connect(self.update_angle_from_box)
        self.angle_slider.valueChanged.connect(self.update_angle_from_slider)

        self.update_parameters()

    def update_angle_from_box(self, value):
        self.theta_deg = float(value)
        self.angle_slider.setValue(int(round(self.theta_deg * 10)))
        self.update_plot()

    def update_angle_from_slider(self, value):
        self.theta_deg = value / 10.0
        self.angle_spin.setValue(self.theta_deg)
        self.update_plot()

    def update_parameters(self):
        self.a1 = float(self.a1_spin.value())
        self.a2 = float(self.a2_spin.value())
        self.rows1 = int(self.rows1_spin.value())
        self.cols1 = int(self.cols1_spin.value())
        self.rows2 = int(self.rows2_spin.value())
        self.cols2 = int(self.cols2_spin.value())
        self.update_plot()

    def update_plot(self):
        self.canvas.plot_lattices(
            self.a1, self.a2, self.theta_deg,
            rows1=self.rows1, cols1=self.cols1,
            rows2=self.rows2, cols2=self.cols2
        )

        Lm = moire_lattice_constant(self.a1, self.a2, self.theta_deg)
        if np.isinf(Lm):
            text = "Twist angle is 0°, so the moiré period is effectively infinite."
        else:
            text = f"θ = {self.theta_deg:.2f}° | Moiré lattice constant = {Lm:.3f} Å"
        self.result_label.setText(text)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())