"""Tests for QMD parsing functionality."""

from smartrappy.qmd_parser import extract_python_chunks


def test_extract_python_chunks():
    """Test that Python chunks are extracted correctly from QMD files."""
    # Sample QMD content with Python chunks
    qmd_content = """# Test QMD File

This is a test QMD file with Python chunks.

```{python}
import pandas as pd
df = pd.read_csv("data.csv")
```

Some markdown text between chunks.

```{python}
df.to_excel("output.xlsx")
```

```{r}
# This is an R chunk that should be ignored
print("Hello from R")
```

```{python}
import matplotlib.pyplot as plt
plt.plot(df["x"], df["y"])
plt.savefig("plot.png")
```
"""

    # Extract Python chunks
    chunks = extract_python_chunks(qmd_content)

    # Check that we found the right number of chunks
    assert len(chunks) == 3

    # Check that the chunks have the right content
    assert "import pandas as pd" in chunks[0]
    assert "df.to_excel(" in chunks[1]
    assert "import matplotlib.pyplot" in chunks[2]

    # Check that the R chunk was ignored
    for chunk in chunks:
        assert "Hello from R" not in chunk


def test_empty_qmd_file():
    """Test handling of QMD files with no Python chunks."""
    qmd_content = """# Empty QMD File

This QMD file has no Python chunks.

```{r}
print("Hello from R")
```
"""
    chunks = extract_python_chunks(qmd_content)
    assert len(chunks) == 0


def test_malformed_chunks():
    """Test handling of malformed Python chunks."""
    qmd_content = """# Malformed QMD File

```{python
# Missing closing brace
x = 1
```

```{python}
# This one is fine
y = 2
```
"""
    # The regex should still handle the malformed chunk
    chunks = extract_python_chunks(qmd_content)
    assert len(chunks) == 1
    assert "y = 2" in chunks[0]


def test_with_metadata():
    """Test handling of Python chunks with metadata."""
    qmd_content = """# QMD with metadata

```{python echo=false, eval=true}
import pandas as pd
df = pd.read_csv("data.csv")
```
"""
    chunks = extract_python_chunks(qmd_content)
    assert len(chunks) == 1
    assert "import pandas as pd" in chunks[0]


def test_with_actual_file(tmp_path):
    """Test extraction from an actual file."""
    # Create a temporary QMD file
    qmd_file = tmp_path / "test.qmd"
    qmd_content = """# Test QMD File

```{python}
import pandas as pd
df = pd.read_csv("data.csv")
df.to_excel("output.xlsx")
```

```{python}
import matplotlib.pyplot as plt
plt.savefig("plot.png")
```
"""
    qmd_file.write_text(qmd_content)

    # Extract chunks from the file
    with open(qmd_file, "r") as f:
        chunks = extract_python_chunks(f.read())

    assert len(chunks) == 2
    assert "import pandas as pd" in chunks[0]
    assert "import matplotlib.pyplot as plt" in chunks[1]
