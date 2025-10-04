import json
import os
import tempfile
from pathlib import Path

from smartrappy import analyse_project
from smartrappy.models import NodeType
from smartrappy.reporters import ConsoleReporter


def test_jupyter_integration():
    """Test that Jupyter notebook files are properly analyzed in a project."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a simple project structure with Python and notebook files
        tmpdir_path = Path(tmpdir)

        # Create a Python file
        py_file = tmpdir_path / "process.py"
        py_file.write_text("""
import pandas as pd

df = pd.read_csv("input.csv")
df.to_excel("output.xlsx")
        """)

        # Create a Jupyter notebook file
        notebook_file = tmpdir_path / "analysis.ipynb"
        notebook_content = {
            "cells": [
                {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": [
                        "# Analysis Notebook\n",
                        "\n",
                        "This is a Jupyter notebook with Python code cells.",
                    ],
                },
                {
                    "cell_type": "code",
                    "execution_count": 1,
                    "metadata": {},
                    "outputs": [],
                    "source": [
                        "import pandas as pd\n",
                        "import matplotlib.pyplot as plt\n",
                        "\n",
                        "df = pd.read_excel('output.xlsx')\n",
                        "plt.plot(df['x'], df['y'])\n",
                        "plt.savefig('plot.png')",
                    ],
                },
                {
                    "cell_type": "code",
                    "execution_count": 2,
                    "metadata": {},
                    "outputs": [],
                    "source": [
                        "# Another code cell\n",
                        "import sqlite3\n",
                        "\n",
                        "conn = sqlite3.connect('data.db')\n",
                        "df_db = pd.read_sql('SELECT * FROM mytable', conn)\n",
                        "df_db.to_csv('db_export.csv')",
                    ],
                },
            ],
            "metadata": {
                "kernelspec": {
                    "display_name": "Python 3",
                    "language": "python",
                    "name": "python3",
                },
                "language_info": {
                    "name": "python",
                    "version": "3.11.0",
                },
            },
            "nbformat": 4,
            "nbformat_minor": 5,
        }
        notebook_file.write_text(json.dumps(notebook_content, indent=2))

        # Create a dummy data file to make it exist on disk
        (tmpdir_path / "input.csv").touch()

        # Analyze the project
        model = analyse_project(str(tmpdir_path))

        # Check that nodes were created for both files
        py_script_found = False
        notebook_found = False

        for node_id, node in model.nodes.items():
            if node.name == "process.py" and node.type == NodeType.SCRIPT:
                py_script_found = True
            elif (
                node.name == "analysis.ipynb" and node.type == NodeType.JUPYTER_NOTEBOOK
            ):
                notebook_found = True

        assert py_script_found, "Python script node not found in the model"
        assert notebook_found, "Jupyter notebook node not found in the model"

        # Check that file operations were detected in the notebook file
        notebook_file_ops = []
        for filename, ops in model.file_operations.items():
            for op in ops:
                if os.path.basename(op.source_file) == "analysis.ipynb":
                    notebook_file_ops.append((filename, op.is_read, op.is_write))

        # Verify expected file operations in the notebook file
        assert ("output.xlsx", True, False) in notebook_file_ops  # Read operation
        assert ("plot.png", False, True) in notebook_file_ops  # Write operation
        assert ("db_export.csv", False, True) in notebook_file_ops  # Write operation

        # Check that database operations were detected
        db_ops_found = False
        for db_name, ops in model.database_operations.items():
            for op in ops:
                if os.path.basename(op.source_file) == "analysis.ipynb":
                    db_ops_found = True
                    break

        assert db_ops_found, "Database operations not found for notebook file"

        # Test that the console reporter can handle notebook files without errors
        reporter = ConsoleReporter()
        reporter.generate_report(model)  # This should not raise exceptions


def test_empty_jupyter_notebook():
    """Test that Jupyter notebooks without code cells are handled correctly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        # Create a notebook file without code cells
        notebook_file = tmpdir_path / "empty.ipynb"
        notebook_content = {
            "cells": [
                {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": [
                        "# Empty Notebook\n",
                        "\n",
                        "This notebook has no code cells.",
                    ],
                },
            ],
            "metadata": {
                "kernelspec": {
                    "display_name": "Python 3",
                    "language": "python",
                    "name": "python3",
                },
            },
            "nbformat": 4,
            "nbformat_minor": 5,
        }
        notebook_file.write_text(json.dumps(notebook_content, indent=2))

        # Analyze the project
        model = analyse_project(str(tmpdir_path))

        # Since there are no code cells, the notebook file should not appear in the model
        notebook_found = False
        for _, node in model.nodes.items():
            if node.name == "empty.ipynb" and node.type == NodeType.JUPYTER_NOTEBOOK:
                notebook_found = True
                break

        assert not notebook_found, "Empty notebook file should not create nodes"


def test_jupyter_integration_with_complex_operations():
    """Test that Jupyter notebooks with complex operations are properly analyzed."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a simple project structure with a notebook containing various operations
        tmpdir_path = Path(tmpdir)

        # Create a Jupyter notebook file with various operations
        notebook_file = tmpdir_path / "analysis.ipynb"
        notebook_content = {
            "cells": [
                {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": [
                        "# Comprehensive Jupyter Notebook\n",
                        "\n",
                        "This notebook includes various types of operations.",
                    ],
                },
                {
                    "cell_type": "code",
                    "execution_count": 1,
                    "metadata": {},
                    "outputs": [],
                    "source": [
                        "import pandas as pd\n",
                        "df = pd.read_csv('data.csv')\n",
                        "df.to_excel('processed_data.xlsx')",
                    ],
                },
                {
                    "cell_type": "code",
                    "execution_count": 2,
                    "metadata": {},
                    "outputs": [],
                    "source": [
                        "import matplotlib.pyplot as plt\n",
                        "plt.figure()\n",
                        "plt.plot(df['x'], df['y'])\n",
                        "plt.savefig('output_plot.png')",
                    ],
                },
                {
                    "cell_type": "code",
                    "execution_count": 3,
                    "metadata": {},
                    "outputs": [],
                    "source": [
                        "# Multiple file operations in one cell\n",
                        "df2 = pd.read_csv('data2.csv')\n",
                        "df3 = pd.read_parquet('data3.parquet')\n",
                        "df_combined = pd.concat([df, df2, df3])\n",
                        "df_combined.to_json('combined.json')",
                    ],
                },
            ],
            "metadata": {
                "kernelspec": {
                    "display_name": "Python 3",
                    "language": "python",
                    "name": "python3",
                },
            },
            "nbformat": 4,
            "nbformat_minor": 5,
        }
        notebook_file.write_text(json.dumps(notebook_content, indent=2))

        # Create input data files
        (tmpdir_path / "data.csv").touch()
        (tmpdir_path / "data2.csv").touch()
        (tmpdir_path / "data3.parquet").touch()

        # Analyze the project
        model = analyse_project(str(tmpdir_path))

        # Check that the notebook was properly processed
        notebook_found = False
        notebook_node_id = None
        for node_id, node in model.nodes.items():
            if node.name == "analysis.ipynb" and node.type == NodeType.JUPYTER_NOTEBOOK:
                notebook_found = True
                notebook_node_id = node_id
                break

        assert notebook_found, "Jupyter notebook node not found in the model"

        # Collect all file operations from the notebook
        notebook_file_ops = []
        for filename, ops in model.file_operations.items():
            for op in ops:
                if os.path.basename(op.source_file) == "analysis.ipynb":
                    notebook_file_ops.append((filename, op.is_read, op.is_write))

        # Verify all operations were detected
        # Read operations
        assert ("data.csv", True, False) in notebook_file_ops
        assert ("data2.csv", True, False) in notebook_file_ops
        assert ("data3.parquet", True, False) in notebook_file_ops

        # Write operations
        assert ("processed_data.xlsx", False, True) in notebook_file_ops
        assert ("output_plot.png", False, True) in notebook_file_ops
        assert ("combined.json", False, True) in notebook_file_ops

        # Verify edges in the graph
        read_edges = 0
        write_edges = 0

        for edge in model.edges:
            if edge.target == notebook_node_id and edge.type == "read":
                read_edges += 1
            elif edge.source == notebook_node_id and edge.type == "write":
                write_edges += 1

        # We should have read edges for the input files
        assert read_edges >= 3, "Not all read operations created edges"
        # We should have write edges for the output files
        assert write_edges >= 3, "Not all write operations created edges"

        # Test that the console reporter works with these operations
        reporter = ConsoleReporter()
        reporter.generate_report(model)  # This should not raise exceptions
