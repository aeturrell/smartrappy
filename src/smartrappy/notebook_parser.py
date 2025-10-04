"""Parser for Jupyter notebook files (.ipynb)."""

import ast
import json
from typing import List, Set, Tuple

from smartrappy.models import DatabaseInfo, FileInfo, ModuleImport


def extract_notebook_cells(notebook_content: str) -> List[str]:
    """
    Extract Python code cells from a Jupyter notebook file.

    Args:
        notebook_content: The content of the .ipynb file as a string

    Returns:
        A list of Python code cell contents found in the notebook
    """
    try:
        # Parse the notebook JSON
        notebook = json.loads(notebook_content)

        # Extract code cells
        code_cells = []

        # Jupyter notebooks have a 'cells' key containing a list of cell objects
        cells = notebook.get("cells", [])

        for cell in cells:
            # Only process cells with type 'code'
            if cell.get("cell_type") == "code":
                # The source can be a string or a list of strings
                source = cell.get("source", [])

                # Convert to a single string
                if isinstance(source, list):
                    cell_code = "".join(source)
                else:
                    cell_code = source

                # Only add non-empty cells
                if cell_code.strip():
                    code_cells.append(cell_code)

        return code_cells

    except (json.JSONDecodeError, KeyError) as e:
        print(f"Error parsing notebook JSON: {str(e)}")
        return []


def analyse_notebook_file(
    file_path: str,
    project_modules: Set[str],
    FileOperationFinder,
    ModuleImportFinder,
    DatabaseOperationFinder,
) -> Tuple[List[FileInfo], List[ModuleImport], List[DatabaseInfo]]:
    """
    Analyse a Jupyter notebook file for Python code cells.

    Args:
        file_path: Path to the .ipynb file
        project_modules: Set of known project module names
        FileOperationFinder: Class to find file operations
        ModuleImportFinder: Class to find module imports
        DatabaseOperationFinder: Class to find database operations

    Returns:
        A tuple of (file_operations, imports, database_operations)
    """
    try:
        # Read the notebook file content
        with open(file_path, "r", encoding="utf-8") as f:
            notebook_content = f.read()

        # Extract Python code cells
        code_cells = extract_notebook_cells(notebook_content)

        # Initialize result lists
        all_file_ops = []
        all_imports = []
        all_db_ops = []

        # Process each code cell separately
        for i, cell_code in enumerate(code_cells):
            try:
                # Parse the cell as Python code
                tree = ast.parse(cell_code)

                # Find file operations
                file_finder = FileOperationFinder(file_path)
                file_finder.visit(tree)
                all_file_ops.extend(file_finder.file_operations)

                # Find imports
                import_finder = ModuleImportFinder(file_path, project_modules)
                import_finder.visit(tree)
                all_imports.extend(import_finder.imports)

                # Find database operations
                db_finder = DatabaseOperationFinder(file_path)
                db_finder.visit(tree)
                all_db_ops.extend(db_finder.database_operations)

            except SyntaxError as e:
                print(f"Syntax error in code cell {i + 1} of {file_path}: {str(e)}")

        return all_file_ops, all_imports, all_db_ops

    except (UnicodeDecodeError, IOError) as e:
        print(f"Error processing notebook file {file_path}: {str(e)}")
        return [], [], []
