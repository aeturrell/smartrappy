import os
import tempfile
from pathlib import Path

from smartrappy import analyse_project
from smartrappy.models import NodeType
from smartrappy.reporters import ConsoleReporter


def test_qmd_integration():
    """Test that QMD files are properly analyzed in a project."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a simple project structure with Python and QMD files
        tmpdir_path = Path(tmpdir)

        # Create a Python file
        py_file = tmpdir_path / "process.py"
        py_file.write_text("""
import pandas as pd

df = pd.read_csv("input.csv")
df.to_excel("output.xlsx")
        """)

        # Create a QMD file
        qmd_file = tmpdir_path / "analysis.qmd"
        qmd_file.write_text("""# Analysis Document

This is a Quarto document with Python code chunks.

```{python}
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_excel("output.xlsx")
plt.plot(df["x"], df["y"])
plt.savefig("plot.png")
```

```{python}
# Another code chunk
import sqlite3

conn = sqlite3.connect("data.db")
df_db = pd.read_sql("SELECT * FROM mytable", conn)
df_db.to_csv("db_export.csv")
```
        """)

        # Create a dummy data file to make it exist on disk
        (tmpdir_path / "input.csv").touch()

        # Analyze the project
        model = analyse_project(str(tmpdir_path))

        # Check that nodes were created for both files
        py_script_found = False
        qmd_doc_found = False

        for node_id, node in model.nodes.items():
            if node.name == "process.py" and node.type == NodeType.SCRIPT:
                py_script_found = True
            elif node.name == "analysis.qmd" and node.type == NodeType.QUARTO_DOCUMENT:
                qmd_doc_found = True

        assert py_script_found, "Python script node not found in the model"
        assert qmd_doc_found, "Quarto document node not found in the model"

        # Check that file operations were detected in the QMD file
        qmd_file_ops = []
        for filename, ops in model.file_operations.items():
            for op in ops:
                if os.path.basename(op.source_file) == "analysis.qmd":
                    qmd_file_ops.append((filename, op.is_read, op.is_write))

        # Verify expected file operations in the QMD file
        assert ("output.xlsx", True, False) in qmd_file_ops  # Read operation
        assert ("plot.png", False, True) in qmd_file_ops  # Write operation
        assert ("db_export.csv", False, True) in qmd_file_ops  # Write operation

        # Check that database operations were detected
        db_ops_found = False
        for db_name, ops in model.database_operations.items():
            for op in ops:
                if os.path.basename(op.source_file) == "analysis.qmd":
                    db_ops_found = True
                    break

        assert db_ops_found, "Database operations not found for QMD file"

        # Test that the console reporter can handle QMD files without errors
        reporter = ConsoleReporter()
        reporter.generate_report(model)  # This should not raise exceptions


def test_empty_qmd():
    """Test that QMD files without Python chunks are handled correctly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        # Create a QMD file without Python chunks
        qmd_file = tmpdir_path / "empty.qmd"
        qmd_file.write_text("""# Empty Document

This Quarto document has no Python code chunks.

```{r}
# R code that should be ignored
print("Hello from R")
```
        """)

        # Analyze the project
        model = analyse_project(str(tmpdir_path))

        # Since there are no Python chunks, the QMD file should not appear in the model
        qmd_found = False
        for _, node in model.nodes.items():
            if node.name == "empty.qmd" and node.type == NodeType.QUARTO_DOCUMENT:
                qmd_found = True
                break

        assert not qmd_found, "Empty QMD file should not create nodes"


def test_qmd_integration_with_all_resources():
    """Test that QMD files with images and include directives are properly analyzed."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a simple project structure with a Quarto document containing various resources
        tmpdir_path = Path(tmpdir)

        # Create a QMD file with markdown images and include directives
        qmd_file = tmpdir_path / "report.qmd"
        qmd_file.write_text("""# Comprehensive Quarto Document

This document includes various types of resources and Python code.

## Images
![First figure](/outputs/figure1.png)

## LaTeX Equation
{{< include /outputs/equation.tex >}}

## Python Analysis
```{python}
import pandas as pd
df = pd.read_csv("data.csv")
df.to_excel("processed_data.xlsx")
```

## Results Visualization
![Results](/outputs/results.svg)

## Data Table
{{< include /outputs/table.html >}}

```{python}
import matplotlib.pyplot as plt
plt.figure()
plt.plot(df["x"], df["y"])
plt.savefig("output_plot.png")
```

## Appendix
{{< include /outputs/appendix.md >}}
        """)

        # Create dummy files to make them exist on disk
        outputs_dir = tmpdir_path / "outputs"
        outputs_dir.mkdir()
        (outputs_dir / "figure1.png").touch()
        (outputs_dir / "results.svg").touch()
        (outputs_dir / "equation.tex").touch()
        (outputs_dir / "table.html").touch()
        (outputs_dir / "appendix.md").touch()

        # Create input data file
        (tmpdir_path / "data.csv").touch()

        # Analyze the project
        model = analyse_project(str(tmpdir_path))

        # Check that the QMD document was properly processed
        qmd_doc_found = False
        qmd_node_id = None
        for node_id, node in model.nodes.items():
            if node.name == "report.qmd" and node.type == NodeType.QUARTO_DOCUMENT:
                qmd_doc_found = True
                qmd_node_id = node_id
                break

        assert qmd_doc_found, "Quarto document node not found in the model"

        # Collect all file operations from the QMD document
        qmd_file_ops = []
        for filename, ops in model.file_operations.items():
            for op in ops:
                if os.path.basename(op.source_file) == "report.qmd":
                    qmd_file_ops.append((filename, op.is_read, op.is_write))

        # Verify all resource types were detected
        # Python code operations
        assert ("data.csv", True, False) in qmd_file_ops  # Read operation
        assert ("processed_data.xlsx", False, True) in qmd_file_ops  # Write operation
        assert ("output_plot.png", False, True) in qmd_file_ops  # Write operation

        # Image references (read operations)
        assert ("outputs/figure1.png", True, False) in qmd_file_ops
        assert ("outputs/results.svg", True, False) in qmd_file_ops

        # Include directives (read operations)
        assert ("outputs/equation.tex", True, False) in qmd_file_ops
        assert ("outputs/table.html", True, False) in qmd_file_ops
        assert ("outputs/appendix.md", True, False) in qmd_file_ops

        # Verify edges in the graph
        image_nodes_with_edges = 0
        include_nodes_with_edges = 0

        for edge in model.edges:
            if edge.target == qmd_node_id and edge.type == "read":
                source_node = model.nodes[edge.source]
                source_name = source_node.name

                # Count image and include nodes with edges to the QMD document
                if source_name in ["outputs/figure1.png", "outputs/results.svg"]:
                    image_nodes_with_edges += 1
                elif source_name in [
                    "outputs/equation.tex",
                    "outputs/table.html",
                    "outputs/appendix.md",
                ]:
                    include_nodes_with_edges += 1

        # Verify we have the right number of edges for each resource type
        assert image_nodes_with_edges == 2, (
            "Not all image nodes have edges to the QMD document"
        )
        assert include_nodes_with_edges == 3, (
            "Not all include nodes have edges to the QMD document"
        )

        # Test that the console reporter works with these resources
        reporter = ConsoleReporter()
        reporter.generate_report(model)  # This should not raise exceptions
