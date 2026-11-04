import csv
import re
import statistics
import numpy as np
from datetime import datetime
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
        writer.writerow(["point", "datetime", "temperature"])
        for point_label, readings in point_readings:
            for timestamp, temperature in readings:
                if row_count % 5 == 0:
                    print(timestamp)
                writer.writerow(
                    [point_label, timestamp.isoformat(sep=" "), temperature]
                )
                row_count += 1

    return row_count


def temperature_stats(readings, start, end):
    """Return start/end temperatures, interval mean, and sample standard deviation."""
    if not readings:
        return None, None, None, None, []

    first_temp = min(readings, key=lambda item: abs(item[0] - start))[1]
    last_temp = min(readings, key=lambda item: abs(item[0] - end))[1]

    interval_readings = [
        (timestamp, temperature)
        for timestamp, temperature in readings
        if start <= timestamp <= end
    ]
    interval_values = [temperature for _, temperature in interval_readings]

    if not interval_values:
        return first_temp, last_temp, None, None, interval_readings

    average_temp = statistics.mean(interval_values)
    temperature_std = (
        statistics.stdev(interval_values)
        if len(interval_values) > 1
        else 0.0
    )

    return first_temp, last_temp, average_temp, temperature_std, interval_readings


def get_point_temperature_ranges(data_dir, temperatures):
    """Get temperature readings within each group's first-to-last .dat time range."""
    groups = {}
    for dat_path in sorted(Path(data_dir).glob("*.dat")):
        run_name = re.sub(r"_\d+$", "", dat_path.stem)
        groups.setdefault(run_name, []).append(dat_path)

    points = []
    for run_name, files in sorted(
        groups.items(),
        key=lambda item: natural_sort_key(item[0]),
    ):
        print("Run: ", run_name)
        dated_files = []
        for dat_path in files:
            try:
                dated_files.append((get_saved_date(dat_path), dat_path))
            except (OSError, ValueError) as error:
                print(f"Skipping {dat_path.name}: {error}")

        if not dated_files:
            continue

        dated_files.sort(key=lambda item: item[0])
        start_time, start_file = dated_files[0]
        end_time, end_file = dated_files[-1]
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


def save_point_temperatures(
    data_dir=None,
    temp_dir=None,
    temp_column=None,
    output_csv=None,
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

    points = get_point_temperature_ranges(data_dir, temperatures)
    if not points:
        messagebox.showerror(
            "No data",
            "No .dat groups with readable Saved Date metadata were found.",
            parent=root,
        )
        root.destroy()
        return

    point_readings = [
        (f"point #{index}", point[-1])
        for index, point in enumerate(points, start=1)
    ]
    try:
        reading_count = save_grouped_temperatures_csv(point_readings, output_csv)
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

    

def main(probabilities=None, tau=None, temp_dir = None):


    root = tk.Tk()
    root.withdraw()

    data_folder = filedialog.askdirectory(
        title="Select folder containing .dat files"
    )
    if not data_folder:
        root.destroy()
        return
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

    output_csv = filedialog.asksaveasfilename(
        title="Save run summary",
        defaultextension=".csv",
        filetypes=[("CSV files", "*.csv")],
        initialfile="run_summary.csv",
    )
    if not output_csv:
        root.destroy()
        return

    print("Loading .dat files from:", data_folder)
    print("Loading temperature records from:", temperature_folder)
    print(f"Saving to: {output_csv}")


    

    temperatures = load_temperatures(
        Path(temperature_folder),
        temp_column,
    )
    if not temperatures:
        messagebox.showerror(
            "No data",
            "No readable temperature records were found.",
            parent=root,
        )
        root.destroy()
        return

    # Group files such as Loop_..._00001.dat and Loop_..._00050.dat
    # under their shared prefix.
    #Sorting the files by name ensures that the first and last files in each group
    print(f"Found {len(list(Path(data_folder).glob('*.dat')))} .dat files in: {data_folder}")
    groups = {}
    for dat_path in sorted(Path(data_folder).glob("*.dat")):
        run_name = re.sub(r"_\d+$", "", dat_path.stem)
        groups.setdefault(run_name, []).append(dat_path)

    if (len(probabilities) != len(groups)):
        print(f"Found {len(groups)} runs in: {data_folder} but {len(probabilities)} probabilities were provided.")
        messagebox.showerror(
            "Probability count mismatch",
            f"There are {len(groups)} runs but "
            f"{len(probabilities)} probabilities.",
            parent=root,
        )
        root.destroy()
        return

    if len(tau) != len(groups):
        print(f"Found {len(groups)} runs in: {data_folder} but {len(tau)} tau values were provided.")
        messagebox.showerror(
            "Tau count mismatch",
            f"There are {len(groups)} runs but "
            f"{len(tau)} tau values.",
            parent=root,
        )
        root.destroy()
        return

    rows = []
    for run_name, files in sorted(groups.items(),key=lambda item: natural_sort_key(item[0]),):
        print(f"Processing run: {run_name}")
        dated_files = []

        for dat_path in files:
            try:
                dated_files.append((get_saved_date(dat_path), dat_path))
            except (OSError, ValueError) as error:
                print(f"Skipping {dat_path.name}: {error}")

        if not dated_files:
            continue

        dated_files.sort(key=lambda item: item[0])
        start_time, start_file = dated_files[0]
        end_time, end_file = dated_files[-1]

        first_temp, last_temp, average_temp, temperature_std, _ = temperature_stats(
            temperatures,
            start_time,
            end_time,
        )

        rows.append({
            "run": run_name,
            "first_file": start_file.name,
            "last_file": end_file.name,
            "start_time": start_time.isoformat(sep=" "),
            "end_time": end_time.isoformat(sep=" "),
            "first_temperature": first_temp,
            "last_temperature": last_temp,
            "average_temperature": average_temp,
            "temperature_std": temperature_std,
        })


    if probabilities is not None:
         if (len(probabilities) != len(rows)) :
            messagebox.showerror(
                "Probability count mismatch",
                f"There are {len(rows)} output rows but "
                f"{len(probabilities)} probabilities.",
                parent=root,
            )
            root.destroy()
            return

    for row, probability in zip(rows, probabilities):
        
        counts, trials, prob = probability
        row["counts"] = counts
        row["trials"] = trials
        row["probability"] = prob
    for row, tau_value in zip(rows, tau):
        row["tau"] = tau_value
    fieldnames = [
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

    try:
        print(f"Writing {len(rows)} run(s) to: {output_csv}")
        with Path(output_csv).open(
            "w",
            newline="",
            encoding="utf-8",
        ) as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        print("Done")
    except OSError as error:
        messagebox.showerror(
            "Save failed",
            str(error),
            parent=root,
        )
        root.destroy()
        return

    messagebox.showinfo(
        "Finished",
        f"Wrote {len(rows)} run(s) to:\n{output_csv}",
        parent=root,
    )
    root.destroy()


if __name__ == "__main__":
    temp_dir = r"E:\Nanoscience data\DBTF\November_2\Temps"
    save_point_temperatures(temp_dir=temp_dir, temp_column=1)


#     counts =        [
#   #D20  # 0,0,0,0,0,0,0,10,25,47,47,49,50,50,50,  50,50,50,50,50,50,50,50,50,50, 50,50,50,50,50,50,50,50,50,50, 50,50,50,50,50,50,50,50,50,50, 50,50,50,50,50,50,50,50,50,50, 50,50,50,50,50,50,50,50,50,50, 50,50,50,50,50,
#     # 0,0,0,0,0,0,0,0,9,30,42,47,50,50,50, 50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50
#               0,0,0,0,0,0,0,0,3,10,27,39,43,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,
#               0,0,0,0,0,0,0,0,0,4,18,31,42,48,49,49,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,34
              
#               ]

#     total_trials =  [
#         # 50,50,50,50,50, 50,50,49,50,50,50,50,50,50,50, 50,50,50,50,50,50,50,50,50,50, 50,50,50,50,50,50,50,50,50,50, 50,50,50,50,50,50,50,50,50,50, 50,50,50,50,50,50,50,50,50,50, 50,50,50,50,50,50,50,50,50,50, 50,50,50,50,50, 
#         # 50,50,50,50,50, 50,50,50,50,50, 50,50,50,50,50, 50,50,50,50,50,50,50,50,50,50, 50,50,50,50,50,50,50,50,50,50,50,50,50,50,50, 50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,
#         50,50,50,50,50,50,50,50,46,44,41,44,46,50,50, 50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,
#         50,50,50,50,50,50,50,50,50,50,48,46,50,50,50, 50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,50,34
        
#         ]

#     probs = [
#     ]

#     tau = [
#          0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,
#          1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1
#        ]

#     print(len(counts), len(total_trials), len(tau))
#     probabilities = []
#     for i, count in enumerate(counts):
        
#         prob = np.asarray(count)/np.asarray(total_trials[i])
#         probabilities.append([count, total_trials[i], prob])
#     main(probabilities, tau, temp_dir=temp_dir)
    