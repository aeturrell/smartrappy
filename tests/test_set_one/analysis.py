from pathlib import Path

import pandas as pd

df = pd.read_csv("data.csv")


def an_example_that_is_imported():
    print("hello")


df_raw_in_data = pd.read_dta(Path("raw/raw_data_in.dta"))
