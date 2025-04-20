import pandas as pd
import sqlite3
import pyodbc
from sqlalchemy import create_engine

# SQLite connection (simplest to test with since it doesn't require a server)
sqlite_conn = sqlite3.connect("example.db")
df1 = pd.read_sql("SELECT * FROM users", sqlite_conn)

# Write to SQLite using pandas
df2 = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})
df2.to_sql("new_table", sqlite_conn, if_exists="replace")

# SQLAlchemy with SQLite
engine = create_engine("sqlite:///another_example.db")
df3 = pd.read_sql_table("some_table", engine)

# MS SQL Server via pyodbc
mssql_conn = pyodbc.connect("DRIVER={SQL Server};SERVER=myserver;DATABASE=mydatabase;UID=user;PWD=password")
df4 = pd.read_sql("SELECT TOP 10 * FROM customers", mssql_conn)

# MS SQL Server via SQLAlchemy
mssql_engine = create_engine("mssql+pyodbc://user:password@myserver/mydatabase?driver=SQL+Server")
df5 = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]})
df5.to_sql("new_mssql_table", mssql_engine, if_exists="replace")

# PostgreSQL via SQLAlchemy
pg_engine = create_engine("postgresql://user:password@localhost:5432/mydatabase")
df6 = pd.read_sql_query("SELECT * FROM pg_tables", pg_engine)
