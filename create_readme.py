import re
from pathlib import Path

import nbformat


def convert_notebook_to_markdown(
    notebook_path: Path, output_path: Path, num_cells: int = 5
) -> None:
    """Converts a Jupyter notebook to a markdown file, including only the first N cells.

    This function reads a Jupyter notebook, extracts a specified number of cells (default 5),
    and converts them to markdown format. Markdown cells are preserved as-is, while code cells
    are wrapped in Python code blocks.

        notebook_path (Path): Path to the input Jupyter notebook file (.ipynb)
        output_path (Path): Path where the output markdown file will be saved
        num_cells (int, optional): Number of cells to include from the start of the notebook. Defaults to 5.

    Returns:
        None: The function prints a confirmation message but does not return any value

    Notes:
        - The function creates the output directory if it doesn't exist
        - Code cells are wrapped in ```python blocks
        - Non-markdown and non-code cells are replaced with HTML comments
    """
    # Load the notebook
    with open(notebook_path, "r", encoding="utf-8") as f:
        notebook = nbformat.read(f, as_version=4)

    # Get the first `num_cells` cells
    cells = notebook.cells[:num_cells]

    # Convert cells to markdown text
    md_lines = []
    for cell in cells:
        if cell.cell_type == "markdown":
            md_lines.append(cell.source)
        elif cell.cell_type == "code":
            md_lines.append("```python\n" + cell.source + "\n```")
        else:
            md_lines.append(f"<!-- Skipped cell of type {cell.cell_type} -->")

    # Join the lines
    markdown_text = "\n\n".join(md_lines)

    # Strip extraneous.
    # Remove special frontmatter
    markdown_text = markdown_text.replace("---\nexecute:\n  echo: false\n---\n", "")

    # Remove width specifications
    markdown_text = re.sub(r"{width=\d+%}", "", markdown_text)

    # Remove leading whitespace and newlines before first hash
    markdown_text = re.sub(r"^\s*(?=#)", "", markdown_text)

    # Replace logo.svg with docs/logo.svg
    markdown_text = markdown_text.replace("logo.svg", "docs/logo.svg")

    # Write to output markdown file
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(markdown_text)

    print(f"Markdown saved to: {output_path}")


if __name__ == "__main__":
    # Example usage
    convert_notebook_to_markdown(
        Path("docs/index.ipynb"), Path("README.md"), num_cells=3
    )
