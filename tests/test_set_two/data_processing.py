import numpy as np
import pandas as pd


def process_data():
    df = pd.read_csv("data/input.csv")
    df["processed"] = df["value"].apply(np.sqrt)
    df.to_csv("data/processed.csv")
