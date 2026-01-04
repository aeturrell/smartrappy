"""Comprehensive tests for analyser.py functions."""

import ast

from smartrappy.analyser import (
    DatabaseOperationFinder,
    FileOperationFinder,
    ModuleImportFinder,
    extract_string_from_node,
    get_direct_db_driver_info,
    get_matplotlib_file_info,
    get_mode_properties,
    get_open_file_info,
    get_pandas_file_info,
    get_pandas_sql_info,
    get_sqlalchemy_info,
)


def parse_call(code: str) -> ast.Call:
    """Parse code containing a single call expression and return the Call node.

    This helper properly narrows the AST type for the type checker.
    """
    tree = ast.parse(code)
    stmt = tree.body[0]
    assert isinstance(stmt, ast.Expr)
    call = stmt.value
    assert isinstance(call, ast.Call)
    return call


class TestGetModeProperties:
    """Test file mode parsing."""

    def test_default_read_mode(self):
        """Test default mode is read-only."""
        is_read, is_write = get_mode_properties(None)
        assert is_read is True
        assert is_write is False

    def test_read_mode(self):
        """Test 'r' mode is read-only."""
        is_read, is_write = get_mode_properties("r")
        assert is_read is True
        assert is_write is False

    def test_write_mode(self):
        """Test 'w' mode is write-only."""
        is_read, is_write = get_mode_properties("w")
        assert is_read is False
        assert is_write is True

    def test_append_mode(self):
        """Test 'a' mode is write-only."""
        is_read, is_write = get_mode_properties("a")
        assert is_read is False
        assert is_write is True

    def test_exclusive_creation_mode(self):
        """Test 'x' mode is write-only."""
        is_read, is_write = get_mode_properties("x")
        assert is_read is False
        assert is_write is True

    def test_read_write_mode(self):
        """Test 'r+' mode allows both read and write."""
        is_read, is_write = get_mode_properties("r+")
        assert is_read is True
        assert is_write is True

    def test_write_read_mode(self):
        """Test 'w+' mode allows both read and write."""
        is_read, is_write = get_mode_properties("w+")
        assert is_read is True
        assert is_write is True

    def test_append_read_mode(self):
        """Test 'a+' mode allows both read and write."""
        is_read, is_write = get_mode_properties("a+")
        assert is_read is True
        assert is_write is True


class TestExtractStringFromNode:
    """Test string extraction from AST nodes."""

    def test_path_call_with_name(self):
        """Test extraction from Path() call."""
        code = 'Path("test.txt")'
        node = parse_call(code)
        result = extract_string_from_node(node)
        assert result == "test.txt"

    def test_path_call_with_attribute(self):
        """Test extraction from pathlib.Path() call."""
        code = 'pathlib.Path("test.txt")'
        node = parse_call(code)
        result = extract_string_from_node(node)
        assert result == "test.txt"

    def test_non_path_call(self):
        """Test that non-Path calls return None."""
        code = 'other_func("test.txt")'
        node = parse_call(code)
        result = extract_string_from_node(node)
        assert result is None


class TestGetOpenFileInfo:
    """Test extraction of file info from open() calls."""

    def test_open_without_args(self):
        """Test open() without arguments returns None."""
        code = "open()"
        node = parse_call(code)
        result = get_open_file_info(node, "test.py")
        assert result is None

    def test_open_with_path_object(self):
        """Test open() with Path object."""
        code = 'open(Path("test.txt"))'
        node = parse_call(code)
        result = get_open_file_info(node, "test.py")
        assert result is not None
        assert result.filename == "test.txt"
        assert result.is_read is True
        assert result.is_write is False

    def test_open_with_keyword_mode(self):
        """Test open() with mode as keyword argument."""
        code = 'open("test.txt", mode="w")'
        node = parse_call(code)
        result = get_open_file_info(node, "test.py")
        assert result is not None
        assert result.filename == "test.txt"
        assert result.is_read is False
        assert result.is_write is True

    def test_open_with_append_mode(self):
        """Test open() with append mode."""
        code = 'open("test.txt", "a")'
        node = parse_call(code)
        result = get_open_file_info(node, "test.py")
        assert result is not None
        assert result.is_write is True

    def test_open_with_read_write_mode(self):
        """Test open() with r+ mode."""
        code = 'open("test.txt", "r+")'
        node = parse_call(code)
        result = get_open_file_info(node, "test.py")
        assert result is not None
        assert result.is_read is True
        assert result.is_write is True


class TestGetPandasFileInfo:
    """Test extraction of file info from pandas operations."""

    def test_pandas_read_without_args(self):
        """Test pd.read_csv() without arguments returns None."""
        code = "pd.read_csv()"
        node = parse_call(code)
        result = get_pandas_file_info(node, "test.py")
        assert result is None

    def test_pandas_non_file_method(self):
        """Test pandas method that doesn't read/write files."""
        code = "pd.concat([df1, df2])"
        node = parse_call(code)
        result = get_pandas_file_info(node, "test.py")
        assert result is None

    def test_dataframe_to_csv_without_args(self):
        """Test df.to_csv() without arguments returns None."""
        code = "df.to_csv()"
        node = parse_call(code)
        result = get_pandas_file_info(node, "test.py")
        assert result is None

    def test_dataframe_to_sql(self):
        """Test df.to_sql() returns None (database, not file)."""
        code = 'df.to_sql("table", conn)'
        node = parse_call(code)
        result = get_pandas_file_info(node, "test.py")
        assert result is None

    def test_pandas_read_sql(self):
        """Test pd.read_sql() returns None (database, not file)."""
        code = 'pd.read_sql("SELECT * FROM table", conn)'
        node = parse_call(code)
        result = get_pandas_file_info(node, "test.py")
        assert result is None


class TestGetMatplotlibFileInfo:
    """Test extraction of file info from matplotlib operations."""

    def test_savefig_with_path_object(self):
        """Test plt.savefig() with Path object."""
        code = 'plt.savefig(Path("plot.png"))'
        node = parse_call(code)
        result = get_matplotlib_file_info(node, "test.py")
        assert result is not None
        assert result.filename == "plot.png"
        assert result.is_write is True


class TestGetSQLAlchemyInfo:
    """Test extraction of database info from SQLAlchemy operations."""

    def test_create_engine_with_string(self):
        """Test create_engine with connection string."""
        code = 'create_engine("sqlite:///test.db")'
        node = parse_call(code)
        result = get_sqlalchemy_info(node, "test.py")
        assert result is not None
        assert result.db_type == "sqlite"
        assert result.connection_string == "sqlite:///test.db"

    def test_create_engine_postgresql(self):
        """Test create_engine with PostgreSQL connection."""
        code = 'create_engine("postgresql://user:pass@localhost/mydb")'
        node = parse_call(code)
        result = get_sqlalchemy_info(node, "test.py")
        assert result is not None
        assert result.db_type == "postgresql"
        assert result.db_name == "mydb"

    def test_create_engine_mysql(self):
        """Test create_engine with MySQL connection."""
        code = 'create_engine("mysql://user:pass@localhost/mydb")'
        node = parse_call(code)
        result = get_sqlalchemy_info(node, "test.py")
        assert result is not None
        assert result.db_type == "mysql"
        assert result.db_name == "mydb"

    def test_create_engine_mssql(self):
        """Test create_engine with MSSQL connection."""
        code = 'create_engine("mssql://user:pass@localhost/mydb")'
        node = parse_call(code)
        result = get_sqlalchemy_info(node, "test.py")
        assert result is not None
        assert result.db_type == "mssql"


class TestGetPandasSQLInfo:
    """Test extraction of database info from pandas SQL operations."""

    def test_read_sql_with_connection_string(self):
        """Test pd.read_sql with connection string."""
        code = 'pd.read_sql("SELECT * FROM table", con="sqlite:///test.db")'
        node = parse_call(code)
        result = get_pandas_sql_info(node, "test.py")
        assert result is not None
        assert result.db_type == "sqlite"
        assert result.is_read is True

    def test_read_sql_with_postgresql_connection(self):
        """Test pd.read_sql with PostgreSQL connection."""
        code = 'pd.read_sql("SELECT * FROM table", con="postgresql://localhost/mydb")'
        node = parse_call(code)
        result = get_pandas_sql_info(node, "test.py")
        assert result is not None
        assert result.db_type == "postgresql"
        assert result.db_name == "mydb"

    def test_read_sql_with_mysql_connection(self):
        """Test pd.read_sql with MySQL connection."""
        code = 'pd.read_sql("SELECT * FROM table", con="mysql://localhost/mydb")'
        node = parse_call(code)
        result = get_pandas_sql_info(node, "test.py")
        assert result is not None
        assert result.db_type == "mysql"

    def test_read_sql_with_mssql_connection(self):
        """Test pd.read_sql with MSSQL ODBC connection."""
        code = 'pd.read_sql("SELECT * FROM table", con="Driver={SQL Server};Server=localhost;Database=mydb")'
        node = parse_call(code)
        result = get_pandas_sql_info(node, "test.py")
        assert result is not None
        assert result.db_type == "mssql"
        assert result.db_name == "mydb"

    def test_read_sql_with_variable_connection(self):
        """Test pd.read_sql with variable connection."""
        code = 'pd.read_sql("SELECT * FROM table", conn_var)'
        node = parse_call(code)
        result = get_pandas_sql_info(node, "test.py")
        # Should still return a DatabaseInfo but without connection details
        assert result is not None


class TestGetDirectDBDriverInfo:
    """Test extraction of database info from direct database drivers."""

    def test_sqlite3_connect(self):
        """Test sqlite3.connect() call."""
        code = 'sqlite3.connect("test.db")'
        node = parse_call(code)
        result = get_direct_db_driver_info(node, "test.py")
        assert result is not None
        assert result.db_type == "sqlite"
        assert result.db_name == "test.db"

    def test_psycopg2_connect(self):
        """Test psycopg2.connect() call."""
        code = 'psycopg2.connect("dbname=mydb user=postgres")'
        node = parse_call(code)
        result = get_direct_db_driver_info(node, "test.py")
        assert result is not None
        assert result.db_type == "postgresql"

    def test_pymysql_connect(self):
        """Test pymysql.connect() call."""
        code = 'pymysql.connect(host="localhost", database="mydb")'
        node = parse_call(code)
        result = get_direct_db_driver_info(node, "test.py")
        assert result is not None
        assert result.db_type == "mysql"

    def test_pyodbc_connect(self):
        """Test pyodbc.connect() call."""
        code = 'pyodbc.connect("Driver={SQL Server};Server=localhost;Database=mydb")'
        node = parse_call(code)
        result = get_direct_db_driver_info(node, "test.py")
        assert result is not None
        assert result.db_type == "mssql"


class TestDatabaseOperationFinder:
    """Test the DatabaseOperationFinder AST visitor."""

    def test_sqlalchemy_engine_tracking(self):
        """Test that SQLAlchemy engines are tracked correctly."""
        code = """
import sqlalchemy as sa
engine = sa.create_engine("sqlite:///test.db")
df = pd.read_sql("SELECT * FROM table", engine)
"""
        tree = ast.parse(code)
        finder = DatabaseOperationFinder("test.py")
        finder.visit(tree)

        # Should find 2 operations: engine creation and read_sql usage
        assert len(finder.database_operations) >= 1

    def test_direct_create_engine(self):
        """Test direct create_engine call tracking."""
        code = """
from sqlalchemy import create_engine
engine = create_engine("postgresql://localhost/mydb")
"""
        tree = ast.parse(code)
        finder = DatabaseOperationFinder("test.py")
        finder.visit(tree)

        assert len(finder.database_operations) >= 1
        # Check that engine was registered
        assert "engine" in finder.sqlalchemy_engines

    def test_connection_variable_tracking(self):
        """Test that database connections are tracked correctly."""
        code = """
import sqlite3
conn = sqlite3.connect("test.db")
df = pd.read_sql("SELECT * FROM table", conn)
"""
        tree = ast.parse(code)
        finder = DatabaseOperationFinder("test.py")
        finder.visit(tree)

        # Should track the connection
        assert len(finder.database_operations) >= 1

    def test_to_sql_with_engine(self):
        """Test df.to_sql() with SQLAlchemy engine."""
        code = """
import sqlalchemy as sa
engine = sa.create_engine("sqlite:///test.db")
df.to_sql("table_name", engine)
"""
        tree = ast.parse(code)
        finder = DatabaseOperationFinder("test.py")
        finder.visit(tree)

        # Should find both engine creation and to_sql operation
        assert len(finder.database_operations) >= 1

    def test_to_sql_with_connection(self):
        """Test df.to_sql() with database connection."""
        code = """
import sqlite3
conn = sqlite3.connect("test.db")
df.to_sql("table_name", conn)
"""
        tree = ast.parse(code)
        finder = DatabaseOperationFinder("test.py")
        finder.visit(tree)

        assert len(finder.database_operations) >= 1

    def test_to_sql_without_tracked_connection(self):
        """Test df.to_sql() without a tracked connection variable."""
        code = """
df.to_sql("table_name", "sqlite:///test.db")
"""
        tree = ast.parse(code)
        finder = DatabaseOperationFinder("test.py")
        finder.visit(tree)

        # Should still detect the operation
        assert len(finder.database_operations) >= 0


class TestFileOperationFinder:
    """Test the FileOperationFinder AST visitor."""

    def test_multiple_file_operations(self):
        """Test finding multiple file operations in one script."""
        code = """
with open("input.txt", "r") as f:
    data = f.read()

with open("output.txt", "w") as f:
    f.write(data)

df = pd.read_csv("data.csv")
df.to_excel("output.xlsx")
"""
        tree = ast.parse(code)
        finder = FileOperationFinder("test.py")
        finder.visit(tree)

        # Should find all 4 file operations
        assert len(finder.file_operations) >= 4


class TestModuleImportFinder:
    """Test the ModuleImportFinder AST visitor."""

    def test_import_tracking(self):
        """Test that imports are tracked correctly."""
        code = """
import pandas as pd
from pathlib import Path
import numpy
"""
        tree = ast.parse(code)
        # ModuleImportFinder requires project_modules parameter
        finder = ModuleImportFinder("test.py", project_modules=set())
        finder.visit(tree)

        # Should find all imports
        assert len(finder.imports) >= 3
