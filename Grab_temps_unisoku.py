import csv
import re
import statistics
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog


def parse_datetime(value):
    value = value.strip().strip('"')

    try:
        return datetime.fromisoformat(value)
    except ValueError:
        for fmt in ("%d.%m.%Y %H:%M:%S", "%d/%m/%Y %H:%M:%S"):
            try:
                return datetime.strptime(value, fmt)
            except ValueError:
                pass

    raise ValueError(f"Unrecognized date/time: {value!r}")


def get_saved_date(dat_path):
    """Find the 'Saved Date' value in a .dat file's metadata."""
    with dat_path.open("r", encoding="utf-8", errors="replace") as file:
        for line in file:
            if line.strip().upper() == "[DATA]":
                break

            match = re.match(r"^\s*Saved Date\s*[\t,]\s*(.*?)\s*$", line, re.I)
            if match:
                return parse_datetime(match.group(1))

    raise ValueError("No 'Saved Date' metadata found")

def natural_sort_key(value):
    """Sort embedded numbers numerically, e.g. run_2 before run_10."""
    return tuple(
        (1, int(part)) if part.isdigit() else (0, part.casefold())
        for part in re.split(r"(\d+)", str(value))
    )


def load_temperatures(temp_dir, temp_column):
    """Read comma-separated temperature log rows with ISO datetimes."""
    readings = []

    for path in sorted(temp_dir.glob("*.txt")):
        with path.open("r", encoding="utf-8-sig", errors="replace") as file:
            for line in file:
                fields = [field.strip() for field in line.strip().split(",")]

                if len(fields) <= temp_column:
                    continue

                try:
                    timestamp = datetime.fromisoformat(fields[0])
                    temperature = float(fields[temp_column])
                except (ValueError, IndexError):
                    # Ignore preambles, headers, and malformed rows.
                    continue

                readings.append((timestamp, temperature))

    return sorted(readings)


def save_temperature_range_csv(temp_dir, temp_column, output_csv):
    """Save every readable temperature record from a folder to a CSV."""
    readings = load_temperatures(Path(temp_dir), temp_column)
    if not readings:
        raise ValueError("No readable temperature records were found.")

    with Path(output_csv).open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["timestamp", "temperature"])
        writer.writerows(
            (timestamp.isoformat(sep=" "), temperature)
            for timestamp, temperature in readings
        )

    return len(readings)


def save_grouped_temperatures_csv(point_readings, output_csv):
    """Save each point's in-range temperature readings to a CSV."""
    row_count = 0
    with Path(output_csv).open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["point", "datetime", "Mean temperature", "Std dev of temperature"])
        for point_label, readings in point_readings:
            for timestamp, temperature, std_dev in readings:
                # if row_count % 5 == 0:
                #     print(timestamp)
                writer.writerow(
                    [point_label, timestamp.isoformat(sep=" "), temperature, std_dev]
                )
                row_count += 1

    return row_count

def average_temperature_spacing_s(temperatures):
    timestamps = sorted({timestamp for timestamp, _ in temperatures})
    if len(timestamps) < 2:
        raise ValueError(
            "At least two distinct temperature timestamps are needed "
            "to calculate the average spacing."
        )

    intervals = [
        (later - earlier).total_seconds()
        for earlier, later in zip(timestamps, timestamps[1:])
    ]
    return statistics.mean(intervals)



def get_point_temperature_ranges(data_dir, temperatures, files_per_group=1, available_time_spacing_s=None):
    """Get temperature readings within each group's first-to-last .dat time range."""
    if files_per_group < 1:
        raise ValueError("Files per group must be at least one.")

    dat_paths = sorted(
        Path(data_dir).glob("*.dat"),
        key=lambda path: natural_sort_key(path.stem),
    )
    print(f"Found {len(dat_paths)} .dat files in {data_dir}")
    # for i, dat_path in enumerate(dat_paths):
    #     if i % 5 == 0:
    #         print(dat_path.name)
    groups = {
        f"point #{index + 1}": dat_paths[start:start + files_per_group]
        for index, start in enumerate(range(0, len(dat_paths), files_per_group))
    }
    for run_name, files in groups.items():
        print(f"{run_name}: {len(files)} files, file 1: {files[0].name}")
    points = []

    if files_per_group == 1:
        spacing_s = (
            available_time_spacing_s
            if available_time_spacing_s is not None
            else average_temperature_spacing_s(temperatures)
        )
        if spacing_s <= 0:
            raise ValueError("Available time spacing must be greater than zero.")

    for run_name, files in sorted(
        groups.items(),
        key=lambda item: natural_sort_key(item[0]),
    ):
        # print("Run: ", run_name)
        dated_files = []
        for dat_path in files:
            try:
                dated_files.append((get_saved_date(dat_path), dat_path))
            except (OSError, ValueError) as error:
                print(f"Skipping {dat_path.name}: {error}")

        if not dated_files:
            continue
        # print(f"{run_name}: {len(dated_files)} files with readable Saved Date metadata")
        dated_files.sort(key=lambda item: item[0])
        start_time, start_file = dated_files[0]
        end_time, end_file = dated_files[-1]    
        if len(dated_files) == 1:
            half_window = timedelta(seconds=spacing_s / 2)
            interval_readings = [
                (timestamp, temperature)
                for timestamp, temperature in temperatures
                if start_time - half_window <= timestamp <= start_time + half_window
            ]
        else:
            interval_readings = [
                (timestamp, temperature)
                for timestamp, temperature in temperatures
                if start_time <= timestamp <= end_time
            ]
        points.append(
            (
                run_name,
                start_file,
                end_file,
                start_time,
                end_time,
                interval_readings,
            )
        )

    return points


def summarize_point_temperatures(points):
    """Return one average temperature row per point, including points with no readings."""
    summaries = []
    for run_name, _, _, start_time, end_time, readings in points:
        point_time = start_time + (end_time - start_time) / 2

        temps = np.asarray([temperature for _, temperature in readings])
        average_temperature = np.mean(temps) if temps.size > 0 else ""
        std_dev_temperature = np.std(temps) if temps.size > 0 else ""
        summaries.append(
            (run_name, [(point_time, average_temperature, round(std_dev_temperature, 3))])
        )
    return summaries


def save_point_temperatures(
    data_dir=None,
    temp_dir=None,
    temp_column=None,
    output_csv=None,
    files_per_group=1,
    available_time_spacing_s=None
):
    """Export temperature readings by .dat group without requiring analysis inputs."""
    root = tk.Tk()
    root.withdraw()

    if data_dir is None:
        data_dir = filedialog.askdirectory(
            title="Select folder containing .dat files",
            parent=root,
        )
        if not data_dir:
            root.destroy()
            return

    if temp_dir is None:
        temp_dir = filedialog.askdirectory(
            title="Select folder containing temperature .txt files",
            parent=root,
        )
        if not temp_dir:
            root.destroy()
            return

    if temp_column is None:
        temp_column = simpledialog.askinteger(
            "Temperature column",
            "Temperature column number after the timestamp "
            "(1 = first temperature column):",
            initialvalue=1,
            minvalue=1,
            parent=root,
        )
        if temp_column is None:
            root.destroy()
            return

    if output_csv is None:
        output_csv = filedialog.asksaveasfilename(
            title="Save temperatures by point",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")],
            initialfile="point_temperatures.csv",
            parent=root,
        )
        if not output_csv:
            root.destroy()
            return

    print(f"Data_dir: {data_dir}")
    print(f"Temp dir: {temp_dir}")
    print(f"Output_CSV: {output_csv}")

    temperatures = load_temperatures(Path(temp_dir), temp_column)
    if not temperatures:
        messagebox.showerror(
            "No data",
            "No readable temperature records were found.",
            parent=root,
        )
        root.destroy()
        return

    points = get_point_temperature_ranges(data_dir, temperatures, files_per_group, available_time_spacing_s=available_time_spacing_s)
    if not points:
        messagebox.showerror(
            "No data",
            "No .dat groups with readable Saved Date metadata were found.",
            parent=root,
        )
        root.destroy()
        return

    point_readings = summarize_point_temperatures(points)
    try:
        save_grouped_temperatures_csv(point_readings, output_csv)
    except OSError as error:
        messagebox.showerror("Save failed", str(error), parent=root)
        root.destroy()
        return
    
    print("Done")
    root.destroy()


def save_temp_cooldown(temp_dir=None):
    root = tk.Tk()
    root.withdraw()

    if temp_dir is None:
            temperature_folder = filedialog.askdirectory(
                title="Select folder containing temperature .txt files"
            )
            if not temperature_folder:
                root.destroy()
                return
            else:
                temperature_folder = temp_dir

    temp_column = simpledialog.askinteger(
            "Temperature column",
            "Temperature column number after the timestamp "
            "(1 = first temperature column):",
            initialvalue=1,
            minvalue=1,
            parent=root,
        )
    if temp_column is None:
        root.destroy()
        return

    temp_csv = filedialog.asksaveasfilename(
        title="Save temperature cooldown",
        defaultextension=".csv",
        filetypes=[("CSV files", "*.csv")],
        initialfile="cooldown.csv",
    )
    if not temp_csv:
        root.destroy()
        return

    len_temps = save_temperature_range_csv(temperature_folder, temp_column, temp_csv)
    print("Saved to: ", temp_csv)

    




if __name__ == "__main__":
    temp_dir = r"E:\Nanoscience data\Pb@MnTe\Temps"
    save_point_temperatures(temp_dir=temp_dir, temp_column=1, files_per_group=1, available_time_spacing_s=10)
    