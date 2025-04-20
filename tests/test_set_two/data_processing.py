import numpy as np
import pandas as pd
import pyodbc


def process_data():
    df = pd.read_csv("data/input.csv")
    df["processed"] = df["value"].apply(np.sqrt)
    df.to_csv("data/processed.csv")


mssql_conn = pyodbc.connect(
    "DRIVER={SQL Server};SERVER=myserver;DATABASE=mydatabase;UID=user;PWD=password"
)
df_db = pd.read_sql("SELECT TOP 10 * FROM customers", mssql_conn)
