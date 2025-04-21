"""Tests for QMD parsing functionality."""

from smartrappy.qmd_parser import extract_markdown_resources, extract_python_chunks


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


def test_extract_markdown_resources():
    """Test that markdown resources are extracted correctly from QMD files."""
    # Sample QMD content with both image references and include directives
    qmd_content = """# Test QMD File

This is a test QMD file with markdown image references and includes.

![A simple image](/path/to/image.png)

Some text between resources.

{{< include /outputs/equation.tex >}}

![Image with spaces in path](/outputs/my diagram.svg)

{{< include "/outputs/table.html" >}}

![External image](https://example.com/image.jpg)

{{< include 'outputs/data.csv' >}}

![Relative path without leading slash](outputs/chart.png)
"""

    # Extract markdown resources
    resources = extract_markdown_resources(qmd_content)

    # Check that we found the right resources (excluding external URLs)
    assert len(resources) == 6  # 3 images (excluding external URL) + 3 includes

    # Check image resources
    image_resources = [path for path, type_ in resources if type_ == "image"]
    assert len(image_resources) == 3
    assert "path/to/image.png" in image_resources
    assert "outputs/my diagram.svg" in image_resources
    assert "outputs/chart.png" in image_resources

    # Check include resources
    include_resources = [path for path, type_ in resources if type_ == "include"]
    assert len(include_resources) == 3
    assert "outputs/equation.tex" in include_resources
    assert "outputs/table.html" in include_resources
    assert "outputs/data.csv" in include_resources


def test_complex_quarto_includes():
    """Test handling of complex Quarto include directives."""
    qmd_content = """# Complex cases

Standard include:
{{< include /outputs/equation.tex >}}

Include with options:
{{< include /outputs/report.md echo=true >}}

Include with multiple options:
{{<include /outputs/data.R echo=true eval=false>}}

Include with whitespace:
{{<    include    /outputs/whitespace.txt    >}}
"""

    resources = extract_markdown_resources(qmd_content)

    # Extract just the include paths
    include_paths = [path for path, type_ in resources if type_ == "include"]

    # Check that we found all includes
    assert len(include_paths) == 4
    assert "outputs/equation.tex" in include_paths
    assert "outputs/report.md" in include_paths  # Should strip options
    assert "outputs/data.R" in include_paths
    assert "outputs/whitespace.txt" in include_paths
