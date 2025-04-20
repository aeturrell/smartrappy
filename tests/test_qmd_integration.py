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
        assert ("plot.png", False, True) in qmd_file_ops     # Write operation
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
