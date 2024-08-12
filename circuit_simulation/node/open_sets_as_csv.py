import pandas as pd
import os
import numpy as np


def get_hardware_data_sets(file_name="sets.csv"):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, file_name)
    data_file = pd.read_csv(file_path, delimiter=";", index_col="nickname")
    return data_file.replace(np.nan, None)
