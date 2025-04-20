import matplotlib.pyplot as plt
import pandas as pd
from data_processing import process_data


def create_plots():
    process_data()
    df = pd.read_csv("data/processed.csv")
    plt.plot(df["processed"])
    plt.savefig("output.png")
