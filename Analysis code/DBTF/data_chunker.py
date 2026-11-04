from distance_analysis import Spectrum
import numpy as np
from numpy.polynomial import polynomial as P
import re
import io
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import struct
import datetime
import os
import glob
from pathlib import Path

basepath = r"Analysis code\DBTF\Distance_experiment_30_05_25\DBTF data"

files = [f for f in os.listdir(basepath)]

for file in files:
    if file.endswith('.dat'):
        file_path = os.path.join(basepath, file)
        data = Spectrum(basepath, file, "X (m)")
        Y = data.give_data("Y (m)")[0]
        current, X = data.x, data.y

        print(data.meta.split('\n')[3].split('\t')[1])
       
