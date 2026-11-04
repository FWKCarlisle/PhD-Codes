import matplotlib.pyplot as plt
import numpy as np
import csv
import os
from Up_down_diffusion import Spectrum
import concurrent.futures as cf
import pandas as pd
from datetime import datetime, timedelta

def loop():
    loop = [3.664,3.4583,3.2774,3.1093,2.9592,2.82205,2.70135,2.59055,2.4928,2.4086,
            4.70415,4.71335,4.7088,4.69675,4.6885,4.6834,4.6782,4.6704,
            4.20765,4.2049,4.2072, 4.20335,4.2067, 4.202,4.2005,4.2025,4.2028,4.2028, 4.2045,4.2047,4.21005]
    errors = [0.101, 0.0947,0.0856,0.0805,0.0729,0.06475,0.05725,0.05195,0.0439,0.0397, 0.00195,
              0.00045,0.0011,0.00045,0.0024,0.0009,0.0005,0.0024,0.0235,0.0004,
              0.0004, 0.00045,0.0002,0.0009,0.0005,0.0015,0.0016,0.0013,0.0006,0.0004,0.00005]
    percents = [87.5,95,99,100,95,100,100,100,100,100,
                0,0,0,0,0,0,0,0,
                 4,4,4,0,1,0,0,1,0,2,0,0,0]

    loop_400 = [4.14555,4.03,3.83155,3.6165,3.41875,
                3.23815,3.0729,2.8403,2.7105,2.58775,
                2.4834,2.39115,2.31,2.23855,2.1748]
    errors_400 = [0.02325,0.0861,0.11365,0.1038,0.09375,
                  0.08395,0.078,0.0686,0.0663,0.05705,
                  0.0492,0.04395,0.0375,0.03315,0.0295]
    percents_400 = [21,29,60,86,94,
                    95,99,98,95,100,
                    99,100,99,100,100]
    
    loop_1s_50 = [4.16265,4.0714,4.15505,3.96155,4.13545,3.83595,3.72025,
                  3.6075,3.4993, 3.3888, 3.28645, 3.19265, 3.10315,
                  3.01845, 2.94015, 2.86525, 2.79425, 2.7302, 2.66975,
                  2.6095]
    errors_1s_50 = [0.00345,0.0429,0.00495,0.06445,0.01545, 0.05985,0.05635,
                    0.0511, 0.0554,0.0533,0.05025,0.04425,0.04055,
                    0.04065,0.03795,0.03775,0.03265,0.0305,0.02945,
                    0.028]
    
    percents_1s_50 = [0,0,0,0,0,12,28,
                      48,76,92,94,100,100,
                      100,100,100,100,100,100,
                      100]


    loop_5s_50 = [4.0532,3.88035,3.7287,3.60715,3.4959,
                  3.38355, 3.26785, 3.16095, 3.0538, 2.95015,
                  2.8535, 2.76185, 2.6744, 2.59405, 2.5203,
                  2.4511, 2.3905, 2.3331, 2.28035]
    errors_5s_50 = [0.083,0.08585,0.0621,0.05525,0.0548,
                    0.05465,0.05475,0.05275,0.0519,0.04835,
                    0.0469,0.04335,0.042,0.03675,0.0342,
                    0.031,0.029,0.0273,0.02535]
    percents_5s_50 = [0,6,20,34,44,
                      71,81,90,100,100,
                      100,100,100,100,100,
                      100,100,100,100]
    # print(len(loop), len(percents), len(errors))
    # print(len(loop_5s_50), len(percents_5s_50), len(errors_5s_50))

    
    plt.scatter(loop, percents, color='blue', label="0.1s")
    plt.scatter(loop_400, percents_400, color='red', label="0.4s")
    plt.scatter(loop_1s_50, percents_1s_50, color='green', label="1s")
    plt.scatter(loop_5s_50, percents_5s_50, color='purple', label="5s")

    plt.xlabel("Temp (K)")
    plt.ylabel("No molecule jump observed (%)")
    plt.errorbar(loop, percents, xerr=errors, fmt='o', color='blue', ecolor='lightgray', elinewidth=2, capsize=2)
    plt.errorbar(loop_400, percents_400, xerr=errors_400, fmt='o', color='red', ecolor='lightgray', elinewidth=2, capsize=2)
    plt.errorbar(loop_1s_50, percents_1s_50, xerr=2*np.array(errors_1s_50), fmt='o', color='green', ecolor='lightgray', elinewidth=2, capsize=2)
    plt.errorbar(loop_5s_50, percents_5s_50, xerr=2*np.array(errors_5s_50), fmt='o', color='purple', ecolor='lightgray', elinewidth=2, capsize=2)
    plt.ylim(-2,102)
    plt.title("Distance from surface, 10nm, 0.1s Looping")
    plt.legend()
    plt.show()

    return 1

def find_csv_rows(path, needle, column=None, exact=False, encoding='utf-8'):
    """
    Search CSV for rows containing `needle`.
    - If column is None: search all fields (substring or exact).
    - If column is given: match only that column.
    - exact=True uses exact equality, else substring match.
    Returns list of dicts (header->value) for matching rows.
    """
    matches = []
    with open(path, newline='', encoding=encoding) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if column is None:
                values = row.values()
            else:
                # if requested column not present, skip
                if column not in row:
                    continue
                values = [row[column]]
            if exact:
                if any(v == needle for v in values):
                    matches.append(row)
            else:
                if any(needle in (v or "") for v in values):
                    matches.append(row)
    return matches

def get_temps(times):

    file = r"E:\data_2025-11-23.txt"
    time_temps = []
    for time in times:
        rows = find_csv_rows(file, time)
        time_splits = []
        temp = 0
        for row in rows:
            timedate = row["timestamp"].split("T")[1].split(".")
            time_splits.append(timedate[1])
            main_time = timedate[0]
            temp += float(row["STM"])
        t = 0
        for splits in time_splits:
            t += float(splits)
        
        time_temps.append([f"{main_time}.{t/len(rows)}", temp/len(rows)])

    avg_temp = 0
    for time_temp in time_temps:
        avg_temp += time_temp[1]

    avg_temp = avg_temp / len(time_temps)
    print(f"Average temp: {avg_temp} K, and error: {abs(avg_temp-time_temps[1][1])} K")
    temp_errors = []
    temp_errors.append([avg_temp,abs(avg_temp-time_temps[1][1])])

    return temp_errors

def get_times(dir_path, files, loop_count=50):
    times = []
    count = 0
    print(files)
    for i, file in enumerate(files):
        if i == 0:
            continue 
        file_spec = Spectrum(dir_path, file, channel="Bias (V)")

        meta_split = np.array(file_spec.meta_split)
        ### ChatGPT added code to find "saved date" in metadata in case it changes per run ###
        mask = np.char.find(np.char.lower(meta_split.astype(str)), "saved date") >= 0
        idxs = np.flatnonzero(mask)
        time_idx = int(idxs[0]) if idxs.size else None

        End_time = meta_split[time_idx].split(" ")[2].strip()
        
        if count == 0:
            Start_time = End_time
            # print(f"Start time: {Start_time}, file: {file}")
        if count == loop_count-1:
            times.append([Start_time, End_time])
            count = 0
            # print(f"End time: {End_time}, file: {file}")
            continue
        count += 1
    return times

def _extract_end_time(args):
    """Helper to run in threads: return (filename, end_time_or_None)."""
    dir_path, file = args
    try:
        spec = Spectrum(dir_path, file, channel="Bias (V)")
        meta_split = np.array(spec.meta_split, dtype=object)
        # case-insensitive search for "saved date"
        idx = next((i for i,s in enumerate(meta_split) if "saved date" in str(s).lower()), None)
        if idx is None:
            return file, None
        # meta line expected like "... Saved Date ... 00:00:00.123456"
        parts = str(meta_split[idx]).split()
        # choose last token that looks like a time with colon
        time_token = next((p for p in reversed(parts) if ":" in p), None)
        return file, time_token
    except Exception:
        return file, None
def get_times_GPT(dir_path, loop_count=50, max_workers=8):
    """
    Parallel extraction of start/end times per file.
    Returns list of [Start_time, End_time] grouped by loop_count.
    """
    files = [f for f in os.listdir(dir_path) if (os.path.isfile(os.path.join(dir_path, f)) and f.endswith(".dat"))]
    
    files = sorted(files)
    args = [(dir_path, f) for f in files]
    
    # parallel read of metadata -> end times (keeps order using map)
    end_times_map = {}
    with cf.ThreadPoolExecutor(max_workers=min(max_workers, max(1, os.cpu_count() or 1)*5)) as ex:
        for file, end_time in ex.map(_extract_end_time, args):
            end_times_map[file] = end_time

    times = []
    count = 0
    Start_time = None
    for i, file in enumerate(files):
        End_time = end_times_map.get(file)
        if End_time is None:
            # skip or continue depending on policy
            continue

        if count == 0:
            Start_time = End_time
            try:
                file_start = file.split("K")[1]
            except Exception:
                file_start = file.split("_")[-2] + "_" + file.split("_")[-1]
        if count == loop_count - 1:
            try:
                file_end = file.split("K")[1]
            except Exception:
                file_end = file.split("_")[-2] + "_" + file.split("_")[-1]
            times.append([[file_start,file_end],[Start_time, End_time]])
            count = 0
            continue
        count += 1

    return times
def get_temps_GPT(times, csv_path=r"E:\data_2025-11-23.txt"):
    """
    Read CSV once with pandas, then query by time (HH:MM:SS) quickly.
    `times` is list of [start_time_str, end_time_str] but function expects
    the time part (HH:MM:SS) or timestamp strings; adapt as needed.
    Returns list of [avg_temp, error] pairs.
    """
    # load CSV once
    df = pd.read_csv(csv_path, dtype=str)
    if 'timestamp' not in df.columns:
        raise RuntimeError("CSV missing 'timestamp' column")

    # create time-only column "HH:MM:SS"
    # handle ISO-like "YYYY-MM-DDTHH:MM:SS.ffffff"
    ts = df['timestamp'].astype(str)
    # safe slice: if 'T' present, take part after T and first 8 chars for HH:MM:SS
    timestr = ts.str.split('T').str.get(1).fillna(ts)
    df['time_hms'] = timestr.str.slice(0,8)

    results = []
    for start_end in times:
        start, end = start_end[1]
        # normalize input: if contains 'T' get hh:mm:ss, or assume "HH:MM:SS" already
        def norm_hms(s):
            if s is None:
                return None
            s = str(s)
            if 'T' in s:
                try:
                    return s.split('T',1)[1][:8]
                except Exception:
                    return None
            # if includes fractional seconds (00:00:00.123) slice to seconds
            return s[:8]
        hms_s = norm_hms(start)
        hms_e = norm_hms(end)
        if not hms_s:
            results.append([None, None])
            continue

        sel_s = df[df['time_hms'] == hms_s]
        if sel_s.empty:
            results.append([None, None])
            continue
            
        sel_e = df[df['time_hms'] == hms_e]
        if sel_e.empty:
            results.append([None, None])
            continue
        
        if 'STM' in sel_s.columns:
            vals_s = pd.to_numeric(sel_s['STM'], errors='coerce').dropna()
            avg_temp_s = float(vals_s.mean()) if not vals_s.empty else None

        if 'STM' in sel_e.columns:
            vals_e = pd.to_numeric(sel_e['STM'], errors='coerce').dropna()
            avg_temp_e = float(vals_e.mean()) if not vals_e.empty else None
        
        temp = (avg_temp_s + avg_temp_e) / 2 if (avg_temp_s is not None and avg_temp_e is not None) else None
        error = float(abs(temp - avg_temp_s)) if (temp is not None and avg_temp_s is not None) else None
        results.append([temp, error])
           
    return results

def save_data(dir_path):
    

    files = [f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))]
    temp_data = [f for f in files if f[-3:] == "txt"]
    
    # times = get_times(dir_path, files)
    # time_temps = []
    # for time in times:
    #     temps = get_temps(time)
    #     time_temps.append([time,temps])
    #     print(f"Time: {time}, Temps: {temps}")

    times = get_times_GPT(dir_path)
    temps = get_temps_GPT(times, csv_path=f"{dir_path}/{temp_data[0]}")

    # percentages = [0,100,None,100,100,100,100,4,14,50,66,76,70,72,100,82] #dataset 9
    # percentages = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,12,0,0,0,0,0,0,0] # dataset 7
    # percentages = [0,100,100,100,100,100,100,0,12,52,60,76,80,94,100,100] #dataset 8
    # percentages = [0,26,33,31,42,45,58,49,54,73,65,0,81,79,74,89,83,0,2,5,1,4,4,10,24] # dataset 5
    # percentages = [92,100,91, 92,94,98,98,98,100,99,100] #dataset 10 0.1s
    # percentages = [0,0,0,0,0,0,0,0,0] #dataset 11 0.1s 27th
    # percentages = [0,0,0,0,2,2,2,12,10,25,25,36,36,47,46,42,60,71,76,77,71,76,85,90,80,86,94,86,83,89,88,92,94,85,85,91,85,90,83,92,90,92,93,0,0] #dataset 11 0.1s 28th
    # percentages = [0,0,0,0,0,0,0,0] #dataset 11 0.5s 27th
    # percentages = [0,0,0,0,0,0,0,4,4,29,32,53,60,59,77,80,83,86,93,86,92,96,98,92,96,98,98,98,98,98,89,91,98,96,96,94,98,98,92,98,89,91,96,0,0] #dataset 11 0.5s 28th
    # percentages = [0,0,6,20,26,29,62,62,90,90] #dataset 13 0.1s 1st
    # percentages = [87,95,91,93,100,91,93,96,95,98,96,91,89,98,91,93,98] #dataset 13 0.1s 2nd
    # percentages = [0,0,0,0,8,15,29,46,68,61] #dataset 13 5s 1st
    # percentages = [72,76,91,78,86,94,93,94,88,98,89,90,90,94,98,94,92,91] #dataset 13 5s 2nd
    # percentages = [0,65,89,93,93,100,96,100,100,100,0,5,0,0,0,0,0,2,0,29] #dataset 14 0.01s 2nd
    # percentages = [100,100,100] # dataset 14 0.01s 3rd
    # percentages = [0,70,96,97,97,96,98,100,96,0,0,0,0,0,0,0,8,44] #dataset 14 0.1s 2nd
    # percentages = [98,100,100] #dataset 14 0.1s 3rd
    # percentages = [0,46,88,98,84,88,98,96,100,0,0,0,0,0,0,0,2,22] #dataset 14 1s 2nd
    # percentages = [92,100,100,100] #dataset 14 1s 3rd
    # percentages = [0,0,0,0,0,12,30,48,72,76,0,0,0,0,0,0,0,0,0] #dataset 18 0.1s 4th
    # percentages = [84,96,96,98,98,96,96,100,100,98,100,100] #dataset 18 0.1s 5th
    # percentages = [0,0,0,0,0,0,0,2,16,0,0,0,0,0,0,0,0,0] #dataset 18 1s 4th
    percentages = [48,66,68,76,88,90,90,88,96,98,98,100,96,100,94,98,98,94,100,98,94,100]
    save_link = rf"E:\DBTF\Data_for_diffusion/dataset_18_1_5th.csv"
    if not os.path.exists(save_link):

        with open(save_link, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Example file Name" , files[1]])
            writer.writerow(["Start file", "End file", "Avg Temp (K)", "Temp Error (K)"])
            try:
                for i, temp in enumerate(temps):
                    write_row = [times[i],temp, percentages[i]]
                    writer.writerow(write_row)
            except Exception as e:
                print(f"Error writing data: {e} -> temps length: {len(temps)}, percentages length: {len(percentages)}")
            writer.writerow(["-----------------------------------------------------------"])

    return 1
def main():
    dir_path = r"E:\DBTF\Data_for_diffusion"

    
    # data_path = r"E:\DBTF\24.11.25-bias_change\Analysed\0.5s" #Dataset 9
    # data_path = r"E:\DBTF\24.11.25-bias_change\Analysed\0.5s" 
    # data_path = r"E:\DBTF\22.11.25-Different_times\0.1s" # Dataset 5
    # data_path = r"E:\DBTF\24.11.25-bias_change\Analysed\0.1s" #Dataset 8
    # data_path = r"E:\DBTF\23.11.25_0.1_loops\analysed" #Dataset 7
    # data_path = r"E:\DBTF\Dataset10\0.1" #dataset 10
    # data_path = r"E:\DBTF\Dataset10\0.5" #dataset 10
    # data_path = r"E:\DBTF\Dataset11\0.1\27th" #dataset 11 0.1 27th
    # data_path = r"E:\DBTF\Dataset11\0.1\28th" #dataset 11 0.1 28th 
    # data_path = r"E:\DBTF\Dataset11\0.5\27th" #dataset 11 0.5 27th
    # data_path = r"E:\DBTF\Dataset11\0.5\28th" #dataset 11 0.5 28th
    # data_path = r"E:\DBTF\Dataset13\0.1\1st\analysed" #dataset 13 0.1 1st 
    # data_path = r"E:\DBTF\Dataset13\0.1\2nd\analysed" #dataset 13 0.1 2nd 
    # data_path = r"E:\DBTF\Dataset13\5\1st\analysed" #dataset 13 5s 1st
    # data_path = r"E:\DBTF\Dataset13\5\2nd\Analyse" #dataset 13 5s 2nd
    # data_path = r"E:\DBTF\Dataset14\0.01\2nd" #dataset 14 0.01s 2nd
    # data_path = r"E:\DBTF\Dataset14\0.01\3rd\analysed" #dataset 14 0.01s 3rd
    # data_path = r"E:\DBTF\Dataset14\0.1\2nd" #dataset 14 0.1s 2nd
    # data_path = r"E:\DBTF\Dataset14\0.1\3rd\analysed" #dataset 14 0.1s 3rd
    # data_path = r"E:\DBTF\Dataset14\1\2nd" #dataset 14 1s 2nd
    # data_path = r"E:\DBTF\Dataset14\1\3rd\analysed" #dataset 14 1s 3rd
    # data_path = r"E:\DBTF\Dataset18\0.1\4th" #dataset 18 0.1s 4th
    # data_path = r"E:\DBTF\Dataset18\0.1\5th" #dataset 18 0.1s 4th
    # data_path = r"E:\DBTF\Dataset18\1s\4th" #dataset 18 1s 4th
    data_path = r"E:\DBTF\Dataset18\1s\5th" #dataset 18 1s 5th
    

    save_data(data_path)

    files = [f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))]
    temps = []
    temp_errors = []
    percentages_array = []
    # percentages = [[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,4,9.5,0,0,0,0,0,0,0,0],
    #                [0,100,None,100,100,100,100,0,12,52,60,76,80,94,100,100],
    #                [3,4],
    #                [22,28,59,85,90,93, 99],
    #                [0,4,14,50,66,76,70,72,100,82,100,None,100,100,100,100]]
    
    data_set_1_temp = [3.664,3.4583,3.2774,3.1093,2.9592,2.82205,2.70135,2.59055,2.4928,2.4086]
    data_set_1_error = [0.101, 0.0947,0.0856,0.0805,0.0729,0.06475,0.05725,0.05195,0.0439,0.0397]
    data_set_1_percent = [87.5,95,99,100,95,100,100,100,100,100,]

    data_set_2_temp = [4.14555,4.03,3.83155,3.6165,3.41875,3.23815,3.0729,2.8403,2.7105,2.58775,2.4834,2.39115,2.31,2.23855,2.1748]
    data_set_2_error = [0.02325,0.0861,0.11365,0.1038,0.09375, 0.08395,0.078,0.0686,0.0663,0.05705,0.0492,0.04395,0.0375,0.03315,0.0295]
    data_set_2_percent = [21,29,60,86,94,95,99,98,95,100,99,100,99,100,100]
    
    data_set_3_temp = [4.16265,4.0714,4.15505,3.96155,4.13545,3.83595,3.72025,3.6075,3.4993, 3.3888, 3.28645, 3.19265, 3.10315,3.01845, 2.94015, 2.86525, 2.79425, 2.7302, 2.66975,2.6095]
    data_set_3_error = [0.00345,0.0429,0.00495,0.06445,0.01545, 0.05985,0.05635,0.0511, 0.0554,0.0533,0.05025,0.04425,0.04055,0.04065,0.03795,0.03775,0.03265,0.0305,0.02945,0.028]
    data_set_3_percent = [0,0,0,0,0,12,28,48,76,92,94,100,100,100,100,100,100,100,100,100]


    data_set_4_temp = [4.0532,3.88035,3.7287,3.60715,3.4959,
                  3.38355, 3.26785, 3.16095, 3.0538, 2.95015,
                  2.8535, 2.76185, 2.6744, 2.59405, 2.5203,
                  2.4511, 2.3905, 2.3331, 2.28035]
    data_set_4_error = [0.083,0.08585,0.0621,0.05525,0.0548,
                    0.05465,0.05475,0.05275,0.0519,0.04835,
                    0.0469,0.04335,0.042,0.03675,0.0342,
                    0.031,0.029,0.0273,0.02535]
    data_set_4_percent = [0,6,20,34,44,
                      71,81,90,100,100,
                      100,100,100,100,100,
                      100,100,100,100] 

    old_data_temps = [data_set_1_temp, data_set_2_temp, data_set_3_temp, data_set_4_temp]
    old_data_errors = [data_set_1_error, data_set_2_error, data_set_3_error, data_set_4_error]
    old_data_percents = [data_set_1_percent, data_set_2_percent, data_set_3_percent, data_set_4_percent]
    data_set_labels = ["Dataset_1_0.1", "Dataset_2_0.4", "Dataset_3_1", "Dataset_4_5"]
    for file in files:
        proxy_temp = []
        proxy = []
        percentages_proxy = []
        with open(rf"{dir_path}\{file}", mode='r') as f:
            lines = f.readlines()
            for line in lines:
                if line.strip().startswith('"['):
                    proxy_temp.append(float(line.split('[')[4].split("]")[0].split(",")[0].strip()))
                    proxy.append(float(line.split("[")[4].split("]")[0].split(",")[1].strip()))
                    percentages_proxy.append(float(line.split(",")[6].strip()))
        temps.append(proxy_temp)
        temp_errors.append(proxy)
        percentages_array.append(percentages_proxy)
    for i, temp_1 in enumerate(temps):
        try:
            # print(len(temp_1), len(percentages_array[i]), len(temp_errors[i]))
            plt.scatter(temp_1,percentages_array[i], label=f"{files[i]}")
            plt.errorbar(temp_1, percentages_array[i], xerr=temp_errors[i], fmt='o', ecolor='lightgray', elinewidth=2, capsize=2)
        except Exception as e:
            print(f"Error plotting file {files[i]}: {e}")
            continue
    # for i, temp in enumerate(old_data_temps):
    #     try:
    #         plt.scatter(temp,old_data_percents[i], label=f"{data_set_labels[i]}")
    #         plt.errorbar(temp, old_data_percents[i], xerr=old_data_errors[i], fmt='o', ecolor='lightgray', elinewidth=2, capsize=2)
    #     except Exception as e:
    #         print(f"Error plotting old data set {i+1}: {e}")
    #         continue
    plt.xlabel("Temp (K)")
    plt.ylabel("No molecule jump observed (%)")
    plt.ylim(-2,102)
    plt.title("Distance from surface, 10nm, 0.1s Looping")
    plt.legend()
    plt.show()

    
if __name__ == "__main__":
    main()