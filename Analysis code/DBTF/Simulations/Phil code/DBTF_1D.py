"""PyQt GUI for a one-dimensional molecular diffusion KMC simulation."""

from __future__ import annotations

import csv
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Callable

import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5 import QtCore, QtGui, QtWidgets


BOLTZMANN_MEV_PER_K = 8.617333262e-2


@dataclass(frozen=True)
class CooldownProfile:
    time_s: np.ndarray
    measured_temperature_k: np.ndarray
    fit_time_s: np.ndarray
    fitted_temperature_k: np.ndarray
    residuals_k: np.ndarray


def load_cooldown_profile(path: str) -> CooldownProfile:
    """Load and time-sort datetime/temperature rows, then fit a smooth cubic trend."""
    try:
        with open(path, newline="", encoding="utf-8-sig") as csv_file:
            reader = csv.DictReader(csv_file)
            if reader.fieldnames is None:
                raise ValueError("The cooldown CSV is empty or has no header.")
            columns = {
                name.strip().lower(): name
                for name in reader.fieldnames
                if name is not None
            }
            if "datetime" not in columns or "temperature" not in columns:
                raise ValueError(
                    "The cooldown CSV must contain 'datetime' and 'temperature' columns."
                )

            timestamps: list[datetime] = []
            temperatures: list[float] = []
            for row_number, row in enumerate(reader, start=2):
                try:
                    timestamps.append(
                        datetime.fromisoformat(row[columns["datetime"]].strip())
                    )
                    temperatures.append(float(row[columns["temperature"]]))
                except (AttributeError, TypeError, ValueError) as exc:
                    raise ValueError(
                        f"Invalid datetime or temperature in CSV row {row_number}."
                    ) from exc
    except OSError as exc:
        raise ValueError(f"Could not read cooldown CSV '{path}': {exc}") from exc
    except csv.Error as exc:
        raise ValueError(f"Could not parse cooldown CSV '{path}': {exc}") from exc

    if len(timestamps) < 5:
        raise ValueError(
            "The smoothed cooldown fit requires at least five measurements."
        )

    try:
        ordered_measurements = sorted(zip(timestamps, temperatures))
        timestamps, temperatures = map(list, zip(*ordered_measurements))
        time_s = np.asarray(
            [(timestamp - timestamps[0]).total_seconds() for timestamp in timestamps],
            dtype=float,
        )
    except TypeError as exc:
        raise ValueError(
            "Cooldown timestamps must use a consistent timezone format."
        ) from exc
    measured_temperature_k = np.asarray(temperatures, dtype=float)
    if (
        not np.all(np.isfinite(time_s))
        or not np.all(np.diff(time_s) > 0)
        or not np.all(np.isfinite(measured_temperature_k))
        or np.any(measured_temperature_k <= 0)
    ):
        raise ValueError(
            "Cooldown timestamps must increase strictly and temperatures must be "
            "finite and greater than zero kelvin."
        )

    duration_s = float(time_s[-1])
    normalized_time = time_s / duration_s
    polynomial = np.polynomial.Polynomial.fit(
        normalized_time, measured_temperature_k, deg=3
    )
    fit_time_s = np.linspace(0.0, duration_s, 201)
    fitted_temperature_k = np.asarray(
        polynomial(fit_time_s / duration_s), dtype=float
    )
    residuals_k = measured_temperature_k - polynomial(normalized_time)
    if (
        not np.all(np.isfinite(fitted_temperature_k))
        or np.any(fitted_temperature_k <= 0)
        or not np.all(np.isfinite(residuals_k))
    ):
        raise ValueError(
            "The smoothed cooldown fit produced invalid temperatures; check the CSV data."
        )

    return CooldownProfile(
        time_s=time_s,
        measured_temperature_k=measured_temperature_k,
        fit_time_s=fit_time_s,
        fitted_temperature_k=fitted_temperature_k,
        residuals_k=np.asarray(residuals_k, dtype=float),
    )


@dataclass(frozen=True)
class SimulationConfig:
    molecule_length_nm: float
    lattice_spacing_nm: float
    attempt_frequency_hz: float
    temperatures_k: np.ndarray
    conditions: tuple[tuple[str, float, float], ...]
    cycles: int
    random_seed: int | None = None
    show_live_view: bool = False
    live_temperature_k: float | None = None
    cooldown_profile: CooldownProfile | None = None
    show_cooldown_diagnostics: bool = False


class OneDimensionalKMC:
    """Continuous-time random walk with equal Arrhenius rates to either neighbour."""

    def __init__(self, config: SimulationConfig) -> None:
        self.config = config
        self._validate_config()

    def _validate_config(self) -> None:
        config = self.config
        if not np.isfinite(config.molecule_length_nm) or config.molecule_length_nm <= 0:
            raise ValueError("Molecule length must be greater than zero.")
        if not np.isfinite(config.lattice_spacing_nm) or config.lattice_spacing_nm <= 0:
            raise ValueError("Lattice spacing must be greater than zero.")
        if not config.conditions:
            raise ValueError("Select at least one plot option.")
        if any(barrier < 0 for _, barrier, _ in config.conditions):
            raise ValueError("The energy barrier cannot be negative.")
        if not np.isfinite(config.attempt_frequency_hz) or config.attempt_frequency_hz <= 0:
            raise ValueError("Attempt frequency must be greater than zero.")
        if config.cycles < 1:
            raise ValueError("Number of cycles must be at least one.")
        if config.temperatures_k.size == 0 or np.any(config.temperatures_k <= 0):
            raise ValueError("Temperatures must be greater than zero kelvin.")
        if any(not np.isfinite(time_away) or time_away < 0 for _, _, time_away in config.conditions):
            raise ValueError("Time away cannot be negative.")
        if config.show_live_view and (
            config.live_temperature_k is None
            or not np.isfinite(config.live_temperature_k)
            or config.live_temperature_k <= 0
        ):
            raise ValueError("Live KMC temperature must be greater than zero kelvin.")

    def run(
        self,
        live_update: Callable[[dict[str, float | int | bool | str]], None] | None = None,
    ) -> list[dict[str, float | int | bool | str]]:
        """Return probe-overlap probabilities for each selected condition and temperature.

        For a constant-rate 1D random walk, the numbers of left and right hops
        during a fixed time are independent Poisson variables.
        A molecule is detected if its center is no farther from the probe position
        than half its length.
        """
        rng = np.random.default_rng(self.config.random_seed)
        if self.config.cooldown_profile is not None:
            return self._run_cooldown(rng)

        results: list[dict[str, float | int | bool | str]] = []
        max_poisson_mean = np.iinfo(np.int64).max / 4
        live_preview_pending = live_update is not None

        for temperature in self.config.temperatures_k:
            for mode, barrier_mev, dwell_time_s in self.config.conditions:
                with np.errstate(over="ignore"):
                    rate_per_direction = self.config.attempt_frequency_hz * np.exp(
                        -barrier_mev / (BOLTZMANN_MEV_PER_K * temperature)
                    )
                mean_hops_each_direction = rate_per_direction * dwell_time_s
                if (
                    not np.isfinite(mean_hops_each_direction)
                    or mean_hops_each_direction > max_poisson_mean
                ):
                    raise ValueError(
                        "The expected number of hops is too large to sample. "
                        "Reduce the attempt frequency or time away."
                    )
                if live_preview_pending:
                    self._stream_live_preview(
                        rng,
                        (
                            self.config.live_temperature_k
                            if self.config.live_temperature_k is not None
                            else temperature
                        ),
                        mode,
                        barrier_mev,
                        dwell_time_s,
                        live_update,
                    )
                    live_preview_pending = False
                left_hops = rng.poisson(mean_hops_each_direction, self.config.cycles)
                right_hops = rng.poisson(mean_hops_each_direction, self.config.cycles)
                displacement_nm = (
                    np.abs(right_hops - left_hops) * self.config.lattice_spacing_nm
                )
                detected = displacement_nm <= self.config.molecule_length_nm / 2
                recovered = int(np.count_nonzero(detected))
                probability = recovered / self.config.cycles
                standard_error = np.sqrt(
                    probability * (1.0 - probability) / self.config.cycles
                )
                results.append(
                    {
                        "mode": mode,
                        "temperature_k": float(temperature),
                        "dwell_time_s": dwell_time_s,
                        "barrier_mev": barrier_mev,
                        "recovered": recovered,
                        "cycles": self.config.cycles,
                        "probability": probability,
                        "standard_error": float(standard_error),
                        "molecule_length_nm": self.config.molecule_length_nm,
                        "lattice_spacing_nm": self.config.lattice_spacing_nm,
                        "attempt_frequency_hz": self.config.attempt_frequency_hz,
                    }
                )
        return results

    def _run_cooldown(
        self, rng: np.random.Generator
    ) -> list[dict[str, float | int | bool | str]]:
        profile = self.config.cooldown_profile
        if profile is None:
            raise ValueError("A cooldown profile is required for cooldown simulation.")

        duration_s = float(profile.time_s[-1])
        results: list[dict[str, float | int | bool | str]] = []

        max_poisson_mean = np.iinfo(np.int64).max / 4
        for mode, barrier_mev, dwell_time_s in self.config.conditions:
            if dwell_time_s < 0:
                raise ValueError("Time away cannot be negative.")
            if dwell_time_s == 0:
                cycle_end_times = np.asarray([0.0])
            else:
                cycle_end_times = np.arange(
                    dwell_time_s, duration_s, dwell_time_s, dtype=float
                )
                if cycle_end_times.size == 0 or cycle_end_times[-1] < duration_s:
                    cycle_end_times = np.append(cycle_end_times, duration_s)

            for cycle_end_s in cycle_end_times:
                cycle_start_s = max(0.0, float(cycle_end_s) - dwell_time_s)
                interval_s = float(cycle_end_s) - cycle_start_s
                if interval_s == 0:
                    mean_hops_each_direction = 0.0
                    temperature_k = float(
                        np.interp(
                            cycle_end_s,
                            profile.fit_time_s,
                            profile.fitted_temperature_k,
                        )
                    )
                else:
                    temperature_k = float(
                        np.interp(
                            cycle_end_s,
                            profile.fit_time_s,
                            profile.fitted_temperature_k,
                        )
                    )
                    integration_points = max(
                        2,
                        int(
                            np.ceil(
                                self.config.temperatures_k.size
                                * interval_s
                                / duration_s
                            )
                        )
                        + 1,
                    )
                    sample_times = np.linspace(
                        cycle_start_s, cycle_end_s, integration_points + 1
                    )
                    midpoint_times = (sample_times[:-1] + sample_times[1:]) / 2
                    midpoint_temperatures = np.interp(
                        midpoint_times,
                        profile.fit_time_s,
                        profile.fitted_temperature_k,
                    )
                    with np.errstate(over="ignore"):
                        rates = self.config.attempt_frequency_hz * np.exp(
                            -barrier_mev
                            / (BOLTZMANN_MEV_PER_K * midpoint_temperatures)
                        )
                    mean_hops_each_direction = float(
                        np.sum(rates * np.diff(sample_times))
                    )

                if (
                    not np.isfinite(mean_hops_each_direction)
                    or mean_hops_each_direction > max_poisson_mean
                ):
                    raise ValueError(
                        "The expected number of hops is too large to sample. "
                        "Reduce the attempt frequency or time away."
                    )

                left_hops = rng.poisson(
                    mean_hops_each_direction, self.config.cycles
                )
                right_hops = rng.poisson(
                    mean_hops_each_direction, self.config.cycles
                )
                displacement_nm = (
                    np.abs(right_hops - left_hops)
                    * self.config.lattice_spacing_nm
                )
                detected = displacement_nm <= self.config.molecule_length_nm / 2
                recovered = int(np.count_nonzero(detected))
                probability = recovered / self.config.cycles
                results.append(
                    {
                        "mode": mode,
                        "temperature_k": temperature_k,
                        "time_s": float(cycle_end_s),
                        "dwell_time_s": dwell_time_s,
                        "barrier_mev": barrier_mev,
                        "recovered": recovered,
                        "cycles": self.config.cycles,
                        "probability": probability,
                        "standard_error": float(
                            np.sqrt(
                                probability * (1.0 - probability)
                                / self.config.cycles
                            )
                        ),
                        "molecule_length_nm": self.config.molecule_length_nm,
                        "lattice_spacing_nm": self.config.lattice_spacing_nm,
                        "attempt_frequency_hz": self.config.attempt_frequency_hz,
                    }
                )

        return results

    def _stream_live_preview(
        self,
        rng: np.random.Generator,
        temperature: float,
        mode: str,
        barrier_mev: float,
        dwell_time_s: float,
        live_update: Callable[[dict[str, float | int | bool | str]], None] | None,
    ) -> None:
        if live_update is None:
            return
        rate_per_direction = self.config.attempt_frequency_hz * np.exp(
            -barrier_mev / (BOLTZMANN_MEV_PER_K * temperature)
        )
        frame_count = 80
        times = np.linspace(0.0, dwell_time_s, frame_count + 1)
        displacement_sites = 0
        positions_nm = [0.0]
        for frame_index in range(1, frame_count + 1):
            interval = times[frame_index] - times[frame_index - 1]
            mean_hops = rate_per_direction * interval
            left_hops = int(rng.poisson(mean_hops))
            right_hops = int(rng.poisson(mean_hops))
            displacement_sites += right_hops - left_hops
            positions_nm.append(
                displacement_sites * self.config.lattice_spacing_nm
            )
            detected = (
                abs(positions_nm[-1]) <= self.config.molecule_length_nm / 2
            )
            live_update(
                {
                    "temperature_k": float(temperature),
                    "mode": mode,
                    "barrier_mev": barrier_mev,
                    "dwell_time_s": dwell_time_s,
                    "time_s": float(times[frame_index]),
                    "positions_nm": positions_nm.copy(),
                    "detected": detected,
                    "complete": frame_index == frame_count,
                    "molecule_length_nm": self.config.molecule_length_nm,
                }
            )


class SimulationWorker(QtCore.QObject):
    finished = QtCore.pyqtSignal(object)
    failed = QtCore.pyqtSignal(str)
    progress = QtCore.pyqtSignal(object)

    def __init__(self, config: SimulationConfig) -> None:
        super().__init__()
        self.config = config

    @QtCore.pyqtSlot()
    def run(self) -> None:
        try:
            callback = self._publish_progress if self.config.show_live_view else None
            results = OneDimensionalKMC(self.config).run(live_update=callback)
            self.finished.emit(results)
        except (ValueError, OverflowError) as exc:
            self.failed.emit(str(exc))

    def _publish_progress(
        self, frame: dict[str, float | int | bool | str]
    ) -> None:
        self.progress.emit(frame)
        time.sleep(0.025)


class LiveKMCWindow(QtWidgets.QDialog):
    """Optional live view of one representative KMC trajectory."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Live KMC trajectory preview")
        self.resize(760, 480)

        layout = QtWidgets.QVBoxLayout(self)
        self.status_label = QtWidgets.QLabel("Waiting for the trajectory preview…")
        layout.addWidget(self.status_label)
        self.figure = Figure(figsize=(8, 4))
        self.canvas = FigureCanvas(self.figure)
        self.axes = self.figure.add_subplot(111)
        layout.addWidget(self.canvas, stretch=1)

    @QtCore.pyqtSlot(object)
    def update_frame(self, frame: dict[str, float | int | bool | str]) -> None:
        positions = np.asarray(frame["positions_nm"], dtype=float)
        dwell_time = float(frame["dwell_time_s"])
        current_time = float(frame["time_s"])
        molecule_length = float(frame["molecule_length_nm"])
        temperature = float(frame["temperature_k"])
        barrier = float(frame["barrier_mev"])

        self.axes.clear()
        self.axes.axhspan(
            -molecule_length / 2,
            molecule_length / 2,
            color="tab:green",
            alpha=0.15,
            label="Molecule overlaps probe",
        )
        self.axes.axhline(0.0, color="0.35", linestyle="--", label="Probe position")
        times = np.linspace(0.0, current_time, positions.size)
        self.axes.plot(times, positions, color="tab:blue", linewidth=1.5)
        self.axes.plot(times[-1], positions[-1], "o", color="tab:red")
        self.axes.set_xlim(0.0, dwell_time if dwell_time > 0 else 1.0)
        maximum_excursion = max(
            molecule_length, float(np.max(np.abs(positions))) * 1.2, 0.1
        )
        self.axes.set_ylim(-maximum_excursion, maximum_excursion)
        self.axes.set_xlabel("Time since tip retraction (s)")
        self.axes.set_ylabel("Molecule center relative to probe (nm)")
        self.axes.grid(True, alpha=0.25)
        self.axes.legend(loc="upper left")
        self.axes.set_title("One KMC trajectory; position is updated during diffusion")

        if bool(frame["complete"]):
            status = (
                "Tip returned — molecule found under probe"
                if bool(frame["detected"])
                else "Tip returned — molecule not under probe"
            )
        else:
            status = "Tip retracted — molecule diffusing"
        self.status_label.setText(
            f"{status} | T={temperature:g} K, barrier={barrier:g} meV, "
            f"time={current_time:.3g}/{dwell_time:.3g} s"
        )
        self.canvas.draw_idle()


class PlotWindow(QtWidgets.QDialog):
    """A resizable top-level window for one simulation figure."""

    def __init__(self, title: str, parent: QtWidgets.QWidget) -> None:
        super().__init__(parent, QtCore.Qt.Window)
        self.setWindowTitle(title)
        self.resize(1000, 700)
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose, False)
        layout = QtWidgets.QVBoxLayout(self)
        self.figure = Figure(figsize=(10, 7))
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas, stretch=1)


class KMCMainWindow(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("1D Molecule Diffusion — Kinetic Monte Carlo")
        self.resize(1100, 720)
        self.results: list[dict[str, float | int]] = []
        self._thread: QtCore.QThread | None = None
        self._worker: SimulationWorker | None = None
        self.live_window: LiveKMCWindow | None = None
        self.last_config: SimulationConfig | None = None
        self.axes: list = []
        self.result_plot_window = PlotWindow("Simulation results", self)
        self.cooldown_plot_window = PlotWindow("Cooldown temperature comparison", self)

        root = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(root)
        self.controls_group = self._build_controls()
        layout.addWidget(self.controls_group)

        self.plot_hint = QtWidgets.QLabel(
            "Simulation results and cooldown diagnostics open in separate plot windows."
        )
        self.plot_hint.setAlignment(QtCore.Qt.AlignCenter)
        layout.addWidget(self.plot_hint, stretch=1)

        self.status_label = QtWidgets.QLabel("Ready.")
        layout.addWidget(self.status_label)
        self.setCentralWidget(root)

    def _build_controls(self) -> QtWidgets.QWidget:
        group = QtWidgets.QGroupBox("Simulation parameters")
        form = QtWidgets.QFormLayout(group)

        self.length_spin = self._float_control(0.001, 10000.0, 3, 3)
        self.spacing_spin = self._float_control(0.001, 1000.0, 0.3, 3)
        self.barrier_spin = self._float_control(0.001, 1000.0, 5, 3)
        self.attempt_edit = QtWidgets.QLineEdit("1e12")
        attempt_validator = QtGui.QDoubleValidator(0.0, 1e300, 300, self)
        attempt_validator.setNotation(QtGui.QDoubleValidator.ScientificNotation)
        self.attempt_edit.setValidator(attempt_validator)
        self.temperature_min_spin = self._float_control(1.0, 5.0, 1.0, 2)
        self.temperature_max_spin = self._float_control(1.0, 5.0, 5.0, 2)
        self.temperature_count_spin = self._integer_control(2, 1000, 250)
        self.temperature_count_label = QtWidgets.QLabel("Number of temperatures")
        self.temperature_count_spin.setToolTip(
            "In cooldown mode, controls the numerical resolution used to integrate "
            "the temperature-dependent hop rate."
        )
        self.dwell_time_spin = self._float_control(0.0, 1e12, 1, 12)
        self.time_min_spin = self._float_control(0.0, 1e12, 0.01, 12)
        self.time_max_spin = self._float_control(0.0, 1e12, 5, 12)
        self.time_count_spin = self._integer_control(2, 100, 5)
        self.barrier_min_spin = self._float_control(0.0, 1000.0, 1.0, 3)
        self.barrier_max_spin = self._float_control(0.0, 1000.0, 5.0, 3)
        self.barrier_count_spin = self._integer_control(2, 100, 5)
        self.cycles_spin = self._integer_control(1, 100000, 1000)
        self.seed_spin = self._integer_control(-1, 2147483647, -1)

        form.addRow("Molecule length (nm)", self.length_spin)
        form.addRow("Lattice spacing (nm)", self.spacing_spin)
        form.addRow("Single-run / time-sweep barrier (meV)", self.barrier_spin)
        form.addRow("Attempt frequency (Hz; e.g. 1e12)", self.attempt_edit)
        form.addRow("Minimum temperature (K)", self.temperature_min_spin)
        form.addRow("Maximum temperature (K)", self.temperature_max_spin)
        form.addRow(self.temperature_count_label, self.temperature_count_spin)
        form.addRow("Single-run / barrier-sweep time away (s)", self.dwell_time_spin)

        cooldown_group = QtWidgets.QGroupBox("Optional measured cooldown")
        cooldown_layout = QtWidgets.QGridLayout(cooldown_group)
        self.cooldown_profile_check = QtWidgets.QCheckBox(
            "Use a smoothed temperature profile from CSV"
        )
        self.cooldown_profile_check.setToolTip(
            "Cycles the tip away and back using the measured temperature history "
            "until the CSV cooldown ends."
        )
        self.cooldown_path_edit = QtWidgets.QLineEdit()
        self.cooldown_path_edit.setReadOnly(True)
        self.cooldown_path_edit.setPlaceholderText("Select a CSV with datetime and temperature columns")
        self.cooldown_browse_button = QtWidgets.QPushButton("Choose CSV…")
        self.cooldown_browse_button.clicked.connect(self._choose_cooldown_csv)
        self.cooldown_diagnostics_check = QtWidgets.QCheckBox(
            "Show temperature fit and residuals"
        )
        self.cooldown_diagnostics_check.setChecked(True)
        cooldown_layout.addWidget(self.cooldown_profile_check, 0, 0, 1, 2)
        cooldown_layout.addWidget(self.cooldown_path_edit, 1, 0)
        cooldown_layout.addWidget(self.cooldown_browse_button, 1, 1)
        cooldown_layout.addWidget(self.cooldown_diagnostics_check, 2, 0, 1, 2)
        form.addRow(cooldown_group)

        plot_options = QtWidgets.QGroupBox("Plots to generate")
        options_layout = QtWidgets.QGridLayout(plot_options)
        self.single_plot_check = QtWidgets.QCheckBox(
            "One barrier and one time away"
        )
        self.single_plot_check.setChecked(True)
        self.time_sweep_check = QtWidgets.QCheckBox("Multiple times away")
        self.barrier_sweep_check = QtWidgets.QCheckBox("Multiple energy barriers")
        options_layout.addWidget(self.single_plot_check, 0, 0, 1, 3)
        options_layout.addWidget(self.time_sweep_check, 1, 0)
        options_layout.addWidget(QtWidgets.QLabel("Time min (s)"), 1, 1)
        options_layout.addWidget(self.time_min_spin, 1, 2)
        options_layout.addWidget(QtWidgets.QLabel("Time max (s)"), 1, 3)
        options_layout.addWidget(self.time_max_spin, 1, 4)
        options_layout.addWidget(QtWidgets.QLabel("Points"), 1, 5)
        options_layout.addWidget(self.time_count_spin, 1, 6)
        options_layout.addWidget(self.barrier_sweep_check, 2, 0)
        options_layout.addWidget(QtWidgets.QLabel("Barrier min (meV)"), 2, 1)
        options_layout.addWidget(self.barrier_min_spin, 2, 2)
        options_layout.addWidget(QtWidgets.QLabel("Barrier max (meV)"), 2, 3)
        options_layout.addWidget(self.barrier_max_spin, 2, 4)
        options_layout.addWidget(QtWidgets.QLabel("Points"), 2, 5)
        options_layout.addWidget(self.barrier_count_spin, 2, 6)
        form.addRow(plot_options)

        form.addRow("Cycles per point", self.cycles_spin)
        form.addRow("Random seed (-1 = random)", self.seed_spin)
        self.live_view_check = QtWidgets.QCheckBox(
            "Open a live window showing one KMC trajectory"
        )
        self.live_view_check.setToolTip(
            "Previews one trajectory at the selected live temperature and first selected condition."
        )
        form.addRow(self.live_view_check)
        self.live_temperature_spin = self._float_control(0.01, 1000.0, 2.0, 2)
        form.addRow("Live KMC temperature (K)", self.live_temperature_spin)

        buttons = QtWidgets.QHBoxLayout()
        self.run_button = QtWidgets.QPushButton("Run simulation")
        self.run_button.clicked.connect(self._start_simulation)
        buttons.addWidget(self.run_button)
        self.save_button = QtWidgets.QPushButton("Save results as CSV")
        self.save_button.setEnabled(False)
        self.save_button.clicked.connect(self._save_csv)
        buttons.addWidget(self.save_button)
        form.addRow(buttons)
        self.single_plot_check.toggled.connect(self._update_plot_controls)
        self.time_sweep_check.toggled.connect(self._update_plot_controls)
        self.barrier_sweep_check.toggled.connect(self._update_plot_controls)
        self.live_view_check.toggled.connect(self._update_plot_controls)
        self.cooldown_profile_check.toggled.connect(self._update_plot_controls)
        self._update_plot_controls()
        return group

    def _choose_cooldown_csv(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Choose measured cooldown CSV",
            "",
            "CSV files (*.csv);;All files (*)",
        )
        if path:
            self.cooldown_path_edit.setText(path)

    def _update_plot_controls(self) -> None:
        cooldown = self.cooldown_profile_check.isChecked()
        single = self.single_plot_check.isChecked()
        time_sweep = self.time_sweep_check.isChecked()
        barrier_sweep = self.barrier_sweep_check.isChecked()
        self.temperature_min_spin.setEnabled(not cooldown)
        self.temperature_max_spin.setEnabled(not cooldown)
        self.temperature_count_spin.setEnabled(True)
        self.temperature_count_label.setText(
            "Cooldown integration points"
            if cooldown
            else "Number of temperatures"
        )
        self.single_plot_check.setEnabled(True)
        self.time_sweep_check.setEnabled(True)
        self.barrier_sweep_check.setEnabled(True)
        self.barrier_spin.setEnabled(single or time_sweep)
        self.dwell_time_spin.setEnabled(single or barrier_sweep)
        for control in (
            self.time_min_spin,
            self.time_max_spin,
            self.time_count_spin,
        ):
            control.setEnabled(time_sweep)
        for control in (
            self.barrier_min_spin,
            self.barrier_max_spin,
            self.barrier_count_spin,
        ):
            control.setEnabled(barrier_sweep)
        self.live_view_check.setEnabled(not cooldown)
        self.live_temperature_spin.setEnabled(
            not cooldown and self.live_view_check.isChecked()
        )
        self.cooldown_path_edit.setEnabled(cooldown)
        self.cooldown_browse_button.setEnabled(cooldown)
        self.cooldown_diagnostics_check.setEnabled(cooldown)

    @staticmethod
    def _integer_control(minimum: int, maximum: int, value: int) -> QtWidgets.QSpinBox:
        control = QtWidgets.QSpinBox()
        control.setRange(minimum, maximum)
        control.setValue(value)
        return control

    @staticmethod
    def _float_control(
        minimum: float, maximum: float, value: float, decimals: int
    ) -> QtWidgets.QDoubleSpinBox:
        control = QtWidgets.QDoubleSpinBox()
        control.setRange(minimum, maximum)
        control.setDecimals(decimals)
        control.setValue(value)
        control.setKeyboardTracking(False)
        return control

    def _read_config(self) -> SimulationConfig:
        minimum_temperature = self.temperature_min_spin.value()
        maximum_temperature = self.temperature_max_spin.value()
        try:
            attempt_frequency = float(self.attempt_edit.text())
        except ValueError as exc:
            raise ValueError(
                "Enter a valid attempt frequency, for example 1e12."
            ) from exc
        if maximum_temperature < minimum_temperature:
            raise ValueError("Maximum temperature must be at least the minimum.")
        if not np.isfinite(attempt_frequency) or attempt_frequency <= 0:
            raise ValueError("Attempt frequency must be finite and greater than zero.")

        cooldown_profile = None
        if self.cooldown_profile_check.isChecked():
            path = self.cooldown_path_edit.text().strip()
            if not path:
                raise ValueError("Choose a cooldown CSV before running this mode.")
            cooldown_profile = load_cooldown_profile(path)
        elif maximum_temperature < minimum_temperature:
            raise ValueError("Maximum temperature must be at least the minimum.")

        base_barrier = self.barrier_spin.value()
        base_time = self.dwell_time_spin.value()
        mutable_conditions: list[tuple[str, float, float]] = []
        if self.single_plot_check.isChecked():
            mutable_conditions.append(("single", base_barrier, base_time))
        if self.time_sweep_check.isChecked():
            minimum_time = self.time_min_spin.value()
            maximum_time = self.time_max_spin.value()
            if maximum_time <= minimum_time:
                raise ValueError("Maximum time away must be greater than the minimum.")
            times = np.linspace(
                minimum_time, maximum_time, self.time_count_spin.value()
            )
            mutable_conditions.extend(
                ("time", base_barrier, float(time_away)) for time_away in times
            )
        if self.barrier_sweep_check.isChecked():
            minimum_barrier = self.barrier_min_spin.value()
            maximum_barrier = self.barrier_max_spin.value()
            if maximum_barrier <= minimum_barrier:
                raise ValueError("Maximum barrier must be greater than the minimum.")
            barriers = np.linspace(
                minimum_barrier, maximum_barrier, self.barrier_count_spin.value()
            )
            mutable_conditions.extend(
                ("barrier", float(barrier), base_time) for barrier in barriers
            )
        if not mutable_conditions:
            raise ValueError("Select at least one plot option.")

        conditions = tuple(mutable_conditions)
        if cooldown_profile is not None:
            temperatures = np.linspace(
                cooldown_profile.fitted_temperature_k[0],
                cooldown_profile.fitted_temperature_k[-1],
                self.temperature_count_spin.value(),
            )
        else:
            temperatures = np.linspace(
                minimum_temperature,
                maximum_temperature,
                self.temperature_count_spin.value(),
            )

        seed = self.seed_spin.value()
        return SimulationConfig(
            molecule_length_nm=self.length_spin.value(),
            lattice_spacing_nm=self.spacing_spin.value(),
            attempt_frequency_hz=attempt_frequency,
            temperatures_k=temperatures,
            conditions=conditions,
            cycles=self.cycles_spin.value(),
            random_seed=None if seed == -1 else seed,
            show_live_view=(
                self.live_view_check.isChecked() and cooldown_profile is None
            ),
            live_temperature_k=(
                self.live_temperature_spin.value()
                if self.live_view_check.isChecked() and cooldown_profile is None
                else None
            ),
            cooldown_profile=cooldown_profile,
            show_cooldown_diagnostics=(
                self.cooldown_diagnostics_check.isChecked()
                if cooldown_profile is not None
                else False
            ),
        )

    def _start_simulation(self) -> None:
        try:
            config = self._read_config()
        except ValueError as exc:
            QtWidgets.QMessageBox.warning(self, "Invalid parameters", str(exc))
            return

        self.last_config = config
        self.controls_group.setEnabled(False)
        self.status_label.setText("Simulation running…")
        if config.show_live_view:
            self.live_window = LiveKMCWindow()
            self.live_window.show()
        self._thread = QtCore.QThread(self)
        self._worker = SimulationWorker(config)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        if self.live_window is not None:
            self._worker.progress.connect(self.live_window.update_frame)
        self._worker.finished.connect(self._on_simulation_finished)
        self._worker.failed.connect(self._on_simulation_failed)
        self._worker.finished.connect(self._thread.quit)
        self._worker.failed.connect(self._thread.quit)
        self._thread.finished.connect(self._worker.deleteLater)
        self._thread.finished.connect(self._on_thread_finished)
        self._thread.start()

    @QtCore.pyqtSlot(object)
    def _on_simulation_finished(
        self, results: list[dict[str, float | int | bool | str]]
    ) -> None:
        self.results = results
        self._draw_results()
        self.save_button.setEnabled(True)
        if self.last_config is not None and self.last_config.cooldown_profile is not None:
            self.status_label.setText(
                f"Cooldown complete: {self.cycles_spin.value()} trajectories per point."
            )
        else:
            self.status_label.setText(
                f"Completed {self.cycles_spin.value()} cycles per temperature."
            )

    @QtCore.pyqtSlot(str)
    def _on_simulation_failed(self, message: str) -> None:
        self.status_label.setText("Simulation failed.")
        QtWidgets.QMessageBox.warning(self, "Simulation error", message)

    @QtCore.pyqtSlot()
    def _on_thread_finished(self) -> None:
        self.controls_group.setEnabled(True)
        self._worker = None
        self._thread = None

    def _draw_results(self) -> None:
        if self.last_config is not None and self.last_config.cooldown_profile is not None:
            self._draw_cooldown_results(self.last_config)
            return

        figure = self.result_plot_window.figure
        figure.clear()

        selected_modes = [
            (self.single_plot_check.isChecked(), "single", "Single barrier and time"),
            (self.time_sweep_check.isChecked(), "time", "Multiple times away"),
            (self.barrier_sweep_check.isChecked(), "barrier", "Multiple energy barriers"),
        ]
        modes = [(mode, title) for selected, mode, title in selected_modes if selected]
        self.axes = [
            figure.add_subplot(1, len(modes), index + 1)
            for index in range(len(modes))
        ]

        for axes, (mode, title) in zip(self.axes, modes):
            mode_rows = [row for row in self.results if row["mode"] == mode]
            curve_values = sorted(
                {
                    row["dwell_time_s"] if mode == "time" else row["barrier_mev"]
                    for row in mode_rows
                }
            ) if mode != "single" else [None]

            for value in curve_values:
                rows = [
                    row for row in mode_rows
                    if value is None
                    or (
                        row["dwell_time_s"] == value
                        if mode == "time"
                        else row["barrier_mev"] == value
                    )
                ]
                rows.sort(key=lambda row: row["temperature_k"])
                temperatures = np.asarray(
                    [row["temperature_k"] for row in rows], dtype=float
                )
                probabilities = np.asarray(
                    [row["probability"] for row in rows], dtype=float
                )
                errors = np.asarray(
                    [row["standard_error"] for row in rows], dtype=float
                )
                label = None
                if mode == "time":
                    label = f"{value:g} s"
                elif mode == "barrier":
                    label = f"{value:g} meV"
                axes.errorbar(
                    temperatures,
                    probabilities,
                    yerr=errors,
                    marker="o",
                    capsize=3,
                    label=label,
                )

            axes.set_xlabel("Temperature (K)")
            axes.set_ylim(-0.02, 1.02)
            axes.set_title(title)
            axes.grid(True, alpha=0.25)
            axes.set_ylabel("Probability molecule overlaps probe on return")
            if mode != "single":
                axes.legend(title="Time away" if mode == "time" else "Barrier")

        figure.set_size_inches(max(9, 7 * len(modes)), 6.5)
        figure.tight_layout()
        self.result_plot_window.canvas.draw_idle()
        self.cooldown_plot_window.hide()
        self.result_plot_window.show()
        self.result_plot_window.raise_()
        self.result_plot_window.activateWindow()

    def _draw_cooldown_results(self, config: SimulationConfig) -> None:
        figure = self.result_plot_window.figure
        figure.clear()
        probability_axes = figure.add_subplot(111)
        self.axes = [probability_axes]
        profile = config.cooldown_profile
        if profile is None:
            raise ValueError("Cooldown diagnostics require a loaded temperature profile.")

        for mode, barrier_mev, dwell_time_s in config.conditions:
            rows = [
                row for row in self.results
                if row["mode"] == mode
                and row["barrier_mev"] == barrier_mev
                and row["dwell_time_s"] == dwell_time_s
            ]
            rows.sort(key=lambda row: row["temperature_k"])
            if not rows:
                continue
            label = None
            if mode == "time":
                label = f"{dwell_time_s:g} s away"
            elif mode == "barrier":
                label = f"{barrier_mev:g} meV"
            probability_axes.errorbar(
                [row["temperature_k"] for row in rows],
                [row["probability"] for row in rows],
                yerr=[row["standard_error"] for row in rows],
                marker="o",
                markersize=3,
                capsize=2,
                linewidth=1,
                label=label,
            )
        probability_axes.set_xlabel("Temperature at tip return (K)")
        probability_axes.set_ylabel("Probability molecule overlaps probe on return")
        probability_axes.set_ylim(-0.02, 1.02)
        probability_axes.set_title("Recovery probability during cooldown")
        probability_axes.grid(True, alpha=0.25)
        if any(mode != "single" for mode, _, _ in config.conditions):
            probability_axes.legend()
        figure.set_size_inches(10, 7)
        figure.tight_layout()
        self.result_plot_window.canvas.draw_idle()
        self.result_plot_window.show()
        self.result_plot_window.raise_()
        self.result_plot_window.activateWindow()

        diagnostics = self.cooldown_plot_window
        diagnostics.figure.clear()
        if config.show_cooldown_diagnostics:
            temperature_axes, residual_axes = diagnostics.figure.subplots(
                2, 1, sharex=True, gridspec_kw={"height_ratios": [2, 1]}
            )
            temperature_axes.plot(
                profile.time_s,
                profile.measured_temperature_k,
                "o",
                markersize=3,
                label="Measured CSV temperatures",
            )
            temperature_axes.plot(
                profile.fit_time_s,
                profile.fitted_temperature_k,
                color="tab:orange",
                label="Smoothed cubic trend",
            )
            temperature_axes.set_ylabel("Temperature (K)")
            temperature_axes.set_title("Measured and smoothed cooldown")
            temperature_axes.grid(True, alpha=0.25)
            temperature_axes.legend()

            residual_axes.axhline(0.0, color="0.35", linestyle="--", linewidth=1)
            residual_axes.plot(
                profile.time_s,
                profile.residuals_k,
                "o-",
                markersize=3,
                color="tab:green",
            )
            residual_axes.set_xlabel("Elapsed cooldown time (s)")
            residual_axes.set_ylabel("Measured − fitted (K)")
            residual_axes.set_title("Temperature-fit residuals")
            residual_axes.grid(True, alpha=0.25)
            diagnostics.figure.set_size_inches(10, 7)
            diagnostics.figure.tight_layout()
            diagnostics.canvas.draw_idle()
            diagnostics.show()
            diagnostics.raise_()
        else:
            diagnostics.hide()

    def _save_csv(self) -> None:
        if not self.results:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save simulation results", "kmc_1d_results.csv", "CSV files (*.csv)"
        )
        if not path:
            return
        fields = [
            "mode",
            "temperature_k",
            "time_s",
            "dwell_time_s",
            "barrier_mev",
            "recovered",
            "cycles",
            "probability",
            "standard_error",
            "molecule_length_nm",
            "lattice_spacing_nm",
            "attempt_frequency_hz",
        ]
        try:
            with open(path, "w", newline="", encoding="utf-8") as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=fields)
                writer.writeheader()
                writer.writerows(self.results)
        except OSError as exc:
            QtWidgets.QMessageBox.critical(
                self, "Could not save CSV", f"Could not write the file:\n{exc}"
            )
            return
        self.status_label.setText(f"Results saved to {path}")


def main() -> None:
    app = QtWidgets.QApplication(sys.argv)
    window = KMCMainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()