"""PyQt GUI for a one-dimensional molecular diffusion KMC simulation."""

from __future__ import annotations

import csv
import multiprocessing
import os
import queue
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import dataclass
from datetime import datetime
from multiprocessing.queues import Queue as MultiprocessingQueue
from typing import Callable

import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5 import QtCore, QtGui, QtWidgets


BOLTZMANN_MEV_PER_K = 8.617333262e-2
MIN_EVENTS_FOR_MULTIPROCESSING = 100_000
_PROCESS_PROGRESS_QUEUE: MultiprocessingQueue | None = None


def _initialize_process_worker(progress_queue: MultiprocessingQueue) -> None:
    global _PROCESS_PROGRESS_QUEUE
    _PROCESS_PROGRESS_QUEUE = progress_queue


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
    process_count: int = 1


class OneDimensionalKMC:
    """Event-by-event continuous-time random walk with equal neighbour rates."""

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
        if config.process_count < 1:
            raise ValueError("Number of KMC processes must be at least one.")
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
        hop_update: Callable[[int], None] | None = None,
    ) -> list[dict[str, float | int | bool | str]]:
        """Return probe-overlap probabilities from event-by-event KMC trajectories."""
        rng = np.random.default_rng(self.config.random_seed)
        results: list[dict[str, float | int | bool | str]] = []
        live_preview_pending = live_update is not None
        total_hops = 0
        last_reported_hops = 0
        next_batch_id = 0

        def record_hops(count: int = 1) -> None:
            nonlocal total_hops, last_reported_hops
            total_hops += count
            while hop_update is not None and total_hops - last_reported_hops >= 1000:
                last_reported_hops += 1000
                hop_update(last_reported_hops)

        process_pool: ProcessPoolExecutor | None = None
        progress_queue: MultiprocessingQueue | None = None
        with np.errstate(over="ignore"):
            estimated_hops = sum(
                2
                * self.config.attempt_frequency_hz
                * np.exp(-barrier / (BOLTZMANN_MEV_PER_K * temperature))
                * dwell_time
                * self.config.cycles
                for temperature in self.config.temperatures_k
                for _, barrier, dwell_time in self.config.conditions
            )
        use_processes = (
            self.config.process_count > 1
            and estimated_hops >= MIN_EVENTS_FOR_MULTIPROCESSING
        )
        if use_processes:
            context = multiprocessing.get_context("spawn")
            progress_queue = context.Queue()
            process_pool = ProcessPoolExecutor(
                max_workers=self.config.process_count,
                mp_context=context,
                initializer=_initialize_process_worker,
                initargs=(progress_queue,),
            )

        try:
            for temperature in self.config.temperatures_k:
                for mode, barrier_mev, dwell_time_s in self.config.conditions:
                    with np.errstate(over="ignore"):
                        rate_per_direction = self.config.attempt_frequency_hz * np.exp(
                            -barrier_mev / (BOLTZMANN_MEV_PER_K * temperature)
                        )
                    if (
                        not np.isfinite(rate_per_direction)
                        or rate_per_direction > np.finfo(float).max / 2
                    ):
                        raise ValueError(
                            "The calculated hopping rate is too large. "
                            "Reduce the attempt frequency or energy barrier."
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
                            record_hops,
                        )
                        live_preview_pending = False

                    if process_pool is None or progress_queue is None:
                        recovered = 0
                        for _ in range(self.config.cycles):
                            displacement_sites = self._simulate_trajectory(
                                rng,
                                rate_per_direction,
                                dwell_time_s,
                                on_hop=(
                                    record_hops if hop_update is not None else None
                                ),
                            )
                            displacement_nm = (
                                abs(displacement_sites)
                                * self.config.lattice_spacing_nm
                            )
                            if displacement_nm <= self.config.molecule_length_nm / 2:
                                recovered += 1
                    else:
                        recovered, next_batch_id = self._run_parallel_condition(
                            process_pool,
                            progress_queue,
                            rng,
                            rate_per_direction,
                            dwell_time_s,
                            next_batch_id,
                            record_hops,
                        )

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
        finally:
            if process_pool is not None:
                process_pool.shutdown()
            if progress_queue is not None:
                progress_queue.close()
                progress_queue.join_thread()

        if hop_update is not None and total_hops != last_reported_hops:
            hop_update(total_hops)
        return results

    def _run_parallel_condition(
        self,
        process_pool: ProcessPoolExecutor,
        progress_queue: MultiprocessingQueue,
        rng: np.random.Generator,
        rate_per_direction: float,
        dwell_time_s: float,
        first_batch_id: int,
        hop_update: Callable[[int], None],
    ) -> tuple[int, int]:
        batch_count = self.config.process_count * 4
        batch_size = max(1, (self.config.cycles + batch_count - 1) // batch_count)
        futures = {}
        batch_id = first_batch_id
        for start in range(0, self.config.cycles, batch_size):
            batch_cycles = min(batch_size, self.config.cycles - start)
            seed = int(rng.integers(0, np.iinfo(np.int64).max))
            future = process_pool.submit(
                _simulate_trajectory_batch,
                (
                    batch_id,
                    seed,
                    batch_cycles,
                    rate_per_direction,
                    dwell_time_s,
                    self.config.molecule_length_nm,
                    self.config.lattice_spacing_nm,
                ),
            )
            futures[future] = batch_id
            batch_id += 1

        pending = set(futures)
        completed: set[int] = set()
        reported_by_batch: dict[int, int] = {}
        recovered = 0

        def drain_progress_queue() -> None:
            while True:
                try:
                    queued_batch_id, hop_increment = progress_queue.get_nowait()
                except queue.Empty:
                    return
                if queued_batch_id not in completed:
                    reported_by_batch[queued_batch_id] = (
                        reported_by_batch.get(queued_batch_id, 0) + hop_increment
                    )
                    hop_update(hop_increment)

        while pending:
            drain_progress_queue()
            finished, pending = wait(
                pending, timeout=0.1, return_when=FIRST_COMPLETED
            )
            for future in finished:
                completed_batch_id, batch_recovered, batch_hops = future.result()
                drain_progress_queue()
                unreported_hops = (
                    batch_hops - reported_by_batch.get(completed_batch_id, 0)
                )
                if unreported_hops > 0:
                    hop_update(unreported_hops)
                completed.add(completed_batch_id)
                recovered += batch_recovered

        drain_progress_queue()
        return recovered, batch_id

    @staticmethod
    def _simulate_trajectory(
        rng: np.random.Generator,
        rate_per_direction: float,
        dwell_time_s: float,
        on_frame: Callable[[float, int, bool], None] | None = None,
        on_hop: Callable[[int], None] | None = None,
    ) -> int:
        """Simulate each exponential waiting time and individual left/right hop."""
        total_rate = 2.0 * rate_per_direction
        position_sites = 0
        frame_count = 80 if on_frame is not None else 0
        if not np.isfinite(total_rate):
            raise ValueError("The total hopping rate is too large to simulate.")
        if dwell_time_s <= 0.0 or total_rate <= 0.0:
            if on_frame is not None:
                for frame_index in range(1, frame_count + 1):
                    on_frame(
                        dwell_time_s * frame_index / frame_count,
                        position_sites,
                        frame_index == frame_count,
                    )
            return position_sites

        current_time = 0.0
        next_frame = 1
        hops_since_update = 0

        while total_rate > 0.0:
            waiting_time = rng.exponential(1.0 / total_rate)
            event_time = current_time + waiting_time
            if event_time > dwell_time_s:
                break

            while (
                on_frame is not None
                and next_frame <= frame_count
                and dwell_time_s * next_frame / frame_count < event_time
            ):
                frame_time = dwell_time_s * next_frame / frame_count
                on_frame(
                    frame_time, position_sites, next_frame == frame_count
                )
                next_frame += 1

            current_time = event_time
            position_sites += 1 if rng.integers(2) else -1
            if on_hop is not None:
                hops_since_update += 1
                if hops_since_update == 1000:
                    on_hop(hops_since_update)
                    hops_since_update = 0

            while (
                on_frame is not None
                and next_frame <= frame_count
                and dwell_time_s * next_frame / frame_count <= current_time
            ):
                frame_time = dwell_time_s * next_frame / frame_count
                on_frame(
                    frame_time, position_sites, next_frame == frame_count
                )
                next_frame += 1

        if on_frame is not None:
            while next_frame <= frame_count:
                frame_time = dwell_time_s * next_frame / frame_count
                on_frame(
                    frame_time,
                    position_sites,
                    next_frame == frame_count,
                )
                next_frame += 1

        if on_hop is not None and hops_since_update:
            on_hop(hops_since_update)
        return position_sites

    def _stream_live_preview(
        self,
        rng: np.random.Generator,
        temperature: float,
        mode: str,
        barrier_mev: float,
        dwell_time_s: float,
        live_update: Callable[[dict[str, float | int | bool | str]], None] | None,
        hop_update: Callable[[int], None] | None = None,
    ) -> None:
        if live_update is None:
            return
        rate_per_direction = self.config.attempt_frequency_hz * np.exp(
            -barrier_mev / (BOLTZMANN_MEV_PER_K * temperature)
        )
        positions_nm = [0.0]

        def publish_frame(
            time_s: float, position_sites: int, complete: bool
        ) -> None:
            positions_nm.append(position_sites * self.config.lattice_spacing_nm)
            detected = abs(positions_nm[-1]) <= self.config.molecule_length_nm / 2
            live_update(
                {
                    "temperature_k": float(temperature),
                    "mode": mode,
                    "barrier_mev": barrier_mev,
                    "dwell_time_s": dwell_time_s,
                    "time_s": time_s,
                    "positions_nm": positions_nm.copy(),
                    "detected": detected,
                    "complete": complete,
                    "molecule_length_nm": self.config.molecule_length_nm,
                }
            )

        self._simulate_trajectory(
            rng,
            rate_per_direction,
            dwell_time_s,
            on_frame=publish_frame,
            on_hop=hop_update,
        )


def _simulate_trajectory_batch(
    task: tuple[int, int, int, float, float, float, float],
) -> tuple[int, int, int]:
    (
        batch_id,
        seed,
        cycles,
        rate_per_direction,
        dwell_time_s,
        molecule_length,
        spacing,
    ) = task
    rng = np.random.default_rng(seed)
    recovered = 0
    total_hops = 0

    def report_hops(count: int) -> None:
        nonlocal total_hops
        previous_hops = total_hops
        total_hops += count
        completed_thousands = total_hops // 1000 - previous_hops // 1000
        if completed_thousands and _PROCESS_PROGRESS_QUEUE is not None:
            _PROCESS_PROGRESS_QUEUE.put((batch_id, completed_thousands * 1000))

    for _ in range(cycles):
        displacement_sites = OneDimensionalKMC._simulate_trajectory(
            rng, rate_per_direction, dwell_time_s, on_hop=report_hops
        )
        if abs(displacement_sites) * spacing <= molecule_length / 2:
            recovered += 1
    return batch_id, recovered, total_hops


class SimulationWorker(QtCore.QObject):
    finished = QtCore.pyqtSignal(object)
    failed = QtCore.pyqtSignal(str)
    progress = QtCore.pyqtSignal(object)
    hop_count = QtCore.pyqtSignal(int)

    def __init__(self, config: SimulationConfig) -> None:
        super().__init__()
        self.config = config

    @QtCore.pyqtSlot()
    def run(self) -> None:
        try:
            start_time = datetime.now().astimezone().isoformat(timespec="milliseconds")
            callback = self._publish_progress if self.config.show_live_view else None
            results = OneDimensionalKMC(self.config).run(
                live_update=callback,
                hop_update=self.hop_count.emit,
            )
            end_time = datetime.now().astimezone().isoformat(timespec="milliseconds")
            for result in results:
                result["start_time"] = start_time
                result["end_time"] = end_time
            self.finished.emit(results)
        except (ValueError, OverflowError, OSError, RuntimeError) as exc:
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


class KMCMainWindow(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("1D Molecule Diffusion — Kinetic Monte Carlo")
        self.resize(1100, 720)
        self.results: list[dict[str, float | int | bool | str]] = []
        self._thread: QtCore.QThread | None = None
        self._worker: SimulationWorker | None = None
        self.live_window: LiveKMCWindow | None = None
        self.axes: list = []

        root = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(root)
        self.controls_group = self._build_controls()
        layout.addWidget(self.controls_group)

        self.figure = Figure(figsize=(9, 6))
        self.canvas = FigureCanvas(self.figure)
        self._show_empty_plot()
        layout.addWidget(self.canvas, stretch=1)

        self.status_label = QtWidgets.QLabel("Ready.")
        layout.addWidget(self.status_label)
        hop_progress_layout = QtWidgets.QHBoxLayout()
        self.hop_count_label = QtWidgets.QLabel("Hop events: 0")
        hop_progress_layout.addWidget(self.hop_count_label)
        self.hop_progress_bar = QtWidgets.QProgressBar()
        self.hop_progress_bar.setTextVisible(False)
        self.hop_progress_bar.setRange(0, 1)
        self.hop_progress_bar.setValue(0)
        hop_progress_layout.addWidget(self.hop_progress_bar, stretch=1)
        layout.addLayout(hop_progress_layout)
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
        self.temperature_count_spin = self._integer_control(2, 1000, 50)
        self.dwell_time_spin = self._float_control(0.0, 1e12, 1, 12)
        self.time_min_spin = self._float_control(0.0, 1e12, 0.01, 12)
        self.time_max_spin = self._float_control(0.0, 1e12, 5, 12)
        self.time_count_spin = self._integer_control(2, 100, 5)
        self.barrier_min_spin = self._float_control(0.0, 1000.0, 1.0, 3)
        self.barrier_max_spin = self._float_control(0.0, 1000.0, 5.0, 3)
        self.barrier_count_spin = self._integer_control(2, 100, 5)
        self.cycles_spin = self._integer_control(1, 100000, 50)
        self.seed_spin = self._integer_control(-1, 2147483647, -1)
        cpu_count = os.cpu_count() or 1
        self.process_count_spin = self._integer_control(
            1, cpu_count, min(8, max(1, cpu_count - 1))
        )
        self.process_count_spin.setToolTip(
            "Small runs stay serial to avoid process-startup overhead."
        )

        form.addRow("Molecule length (nm)", self.length_spin)
        form.addRow("Lattice spacing (nm)", self.spacing_spin)
        form.addRow("Single-run / time-sweep barrier (meV)", self.barrier_spin)
        form.addRow("Attempt frequency (Hz; e.g. 1e12)", self.attempt_edit)
        form.addRow("Minimum temperature (K)", self.temperature_min_spin)
        form.addRow("Maximum temperature (K)", self.temperature_max_spin)
        form.addRow("Number of temperatures", self.temperature_count_spin)
        form.addRow("Single-run / barrier-sweep time away (s)", self.dwell_time_spin)

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
        form.addRow("KMC processes (1 = serial)", self.process_count_spin)
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
        self._update_plot_controls()
        return group

    def _update_plot_controls(self) -> None:
        single = self.single_plot_check.isChecked()
        time_sweep = self.time_sweep_check.isChecked()
        barrier_sweep = self.barrier_sweep_check.isChecked()
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
        self.live_temperature_spin.setEnabled(self.live_view_check.isChecked())

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

    def _show_empty_plot(self) -> None:
        self.figure.clear()
        self.axes = [self.figure.add_subplot(111)]
        self.axes[0].text(
            0.5, 0.5, "Select plot options and run the simulation",
            ha="center", va="center", transform=self.axes[0].transAxes,
        )
        self.axes[0].set_axis_off()
        self.canvas.draw_idle()

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

        base_barrier = self.barrier_spin.value()
        base_time = self.dwell_time_spin.value()
        conditions: list[tuple[str, float, float]] = []
        if self.single_plot_check.isChecked():
            conditions.append(("single", base_barrier, base_time))
        if self.time_sweep_check.isChecked():
            minimum_time = self.time_min_spin.value()
            maximum_time = self.time_max_spin.value()
            if maximum_time <= minimum_time:
                raise ValueError("Maximum time away must be greater than the minimum.")
            times = np.linspace(
                minimum_time, maximum_time, self.time_count_spin.value()
            )
            conditions.extend(
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
            conditions.extend(
                ("barrier", float(barrier), base_time) for barrier in barriers
            )
        if not conditions:
            raise ValueError("Select at least one plot option.")

        seed = self.seed_spin.value()
        return SimulationConfig(
            molecule_length_nm=self.length_spin.value(),
            lattice_spacing_nm=self.spacing_spin.value(),
            attempt_frequency_hz=attempt_frequency,
            temperatures_k=np.linspace(
                minimum_temperature,
                maximum_temperature,
                self.temperature_count_spin.value(),
            ),
            conditions=tuple(conditions),
            cycles=self.cycles_spin.value(),
            process_count=self.process_count_spin.value(),
            random_seed=None if seed == -1 else seed,
            show_live_view=self.live_view_check.isChecked(),
            live_temperature_k=(
                self.live_temperature_spin.value()
                if self.live_view_check.isChecked()
                else None
            ),
        )

    def _start_simulation(self) -> None:
        try:
            config = self._read_config()
        except ValueError as exc:
            QtWidgets.QMessageBox.warning(self, "Invalid parameters", str(exc))
            return

        estimated_events = sum(
            2
            * config.attempt_frequency_hz
            * np.exp(-barrier / (BOLTZMANN_MEV_PER_K * temperature))
            * dwell_time
            * config.cycles
            for temperature in config.temperatures_k
            for _, barrier, dwell_time in config.conditions
        )
        if estimated_events >= 1e7:
            response = QtWidgets.QMessageBox.question(
                self,
                "Large KMC run",
                "This event-by-event KMC run is expected to simulate about "
                f"{estimated_events:.2g} hop events and may take a long time. "
                "Continue?",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                QtWidgets.QMessageBox.No,
            )
            if response != QtWidgets.QMessageBox.Yes:
                return

        self.controls_group.setEnabled(False)
        self.status_label.setText("Simulation running…")
        self.hop_count_label.setText("Hop events: 0")
        self.hop_progress_bar.setRange(0, 0)
        if config.show_live_view:
            self.live_window = LiveKMCWindow()
            self.live_window.show()
        self._thread = QtCore.QThread(self)
        self._worker = SimulationWorker(config)
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.hop_count.connect(self._on_hop_count_updated)
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
        self.status_label.setText(
            f"Completed {self.cycles_spin.value()} cycles per temperature."
        )
        self.hop_progress_bar.setRange(0, 1)
        self.hop_progress_bar.setValue(1)

    @QtCore.pyqtSlot(int)
    def _on_hop_count_updated(self, hop_count: int) -> None:
        self.hop_count_label.setText(f"Hop events: {hop_count:,}")

    @QtCore.pyqtSlot(str)
    def _on_simulation_failed(self, message: str) -> None:
        self.status_label.setText("Simulation failed.")
        self.hop_progress_bar.setRange(0, 1)
        self.hop_progress_bar.setValue(0)
        QtWidgets.QMessageBox.warning(self, "Simulation error", message)

    @QtCore.pyqtSlot()
    def _on_thread_finished(self) -> None:
        self.controls_group.setEnabled(True)
        self._worker = None
        self._thread = None

    def _draw_results(self) -> None:
        self.figure.clear()
        if not self.results:
            axes = self.figure.add_subplot(111)
            axes.text(
                0.5, 0.5, "Select plot options and run the simulation",
                ha="center", va="center", transform=axes.transAxes,
            )
            axes.set_axis_off()
            self.axes = [axes]
            self.canvas.draw_idle()
            return

        selected_modes = [
            (self.single_plot_check.isChecked(), "single", "Single barrier and time"),
            (self.time_sweep_check.isChecked(), "time", "Multiple times away"),
            (self.barrier_sweep_check.isChecked(), "barrier", "Multiple energy barriers"),
        ]
        modes = [(mode, title) for selected, mode, title in selected_modes if selected]
        self.axes = [
            self.figure.add_subplot(1, len(modes), index + 1)
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

        self.figure.set_size_inches(max(8, 6 * len(modes)), 5.5)
        self.figure.tight_layout()
        self.canvas.draw_idle()

    def _save_csv(self) -> None:
        if not self.results:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save simulation results",
            r"E:\Nanoscience data\DBTF\November_2\Summary_CSV\D27_summary.csv",
            "CSV files (*.csv)",
        )
        if not path:
            return
        fields = [
            "run",
            "first_file",
            "last_file",
            "start_time",
            "end_time",
            "first_temperature",
            "last_temperature",
            "average_temperature",
            "temperature_std",
            "counts",
            "trials",
            "probability",
            "tau",
        ]
        summary_rows = [
            {
                "run": run,
                "first_file": "None",
                "last_file": "None",
                "start_time": result["start_time"],
                "end_time": result["end_time"],
                "first_temperature": result["temperature_k"],
                "last_temperature": result["temperature_k"],
                "average_temperature": result["temperature_k"],
                "temperature_std": 0.0,
                "counts": result["recovered"],
                "trials": result["cycles"],
                "probability": result["probability"],
                "tau": result["dwell_time_s"],
            }
            for run, result in enumerate(self.results, start=1)
        ]
        try:
            with open(path, "w", newline="", encoding="utf-8") as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=fields)
                writer.writeheader()
                writer.writerows(summary_rows)
        except OSError as exc:
            QtWidgets.QMessageBox.critical(
                self, "Could not save CSV", f"Could not write the file:\n{exc}"
            )
            return
        self.status_label.setText(f"Results saved to {path}")


def main() -> None:
    multiprocessing.freeze_support()
    app = QtWidgets.QApplication(sys.argv)
    window = KMCMainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()