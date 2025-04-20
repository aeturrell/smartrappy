"""Code analyzer for smartrappy."""

import ast
import os
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from smartrappy.models import FileInfo, ModuleImport, ProjectModel


def get_mode_properties(mode: str) -> tuple[bool, bool]:
    """
    Determine read/write properties from a file mode string.

    Args:
        mode: File mode string (e.g., 'r', 'w', 'a', 'x', 'r+', etc.)

    Returns:
        Tuple of (is_read, is_write)
    """
    # Default mode 'r' if not specified
    mode = mode or "r"

    # Plus sign adds read & write capabilities
    if "+" in mode:
        return True, True

    # Basic mode mapping
    mode_map = {
        "r": (True, False),  # read only
        "w": (False, True),  # write only (truncate)
        "a": (False, True),  # write only (append)
        "x": (False, True),  # write only (exclusive creation)
    }

    # Get base mode (first character)
    base_mode = mode[0]
    return mode_map.get(base_mode, (False, False))


def get_open_file_info(node: ast.Call, source_file: str) -> Optional[FileInfo]:
    """Extract file information from an open() function call."""
    if not (isinstance(node.func, ast.Name) and node.func.id == "open"):
        return None

    # Get filename from first argument
    if not (len(node.args) > 0 and isinstance(node.args[0], ast.Str)):
        return None

    filename = node.args[0].s

    # Default mode is 'r'
    mode = "r"

    # Check positional mode argument
    if len(node.args) > 1 and isinstance(node.args[1], ast.Str):
        mode = node.args[1].s

    # Check for mode in keyword arguments
    for keyword in node.keywords:
        if keyword.arg == "mode" and isinstance(keyword.value, ast.Str):
            mode = keyword.value.s

    is_read, is_write = get_mode_properties(mode)

    return FileInfo(
        filename=filename, is_read=is_read, is_write=is_write, source_file=source_file
    )


def get_pandas_file_info(node: ast.Call, source_file: str) -> Optional[FileInfo]:
    """Extract file information from pandas operations (both pd.read_* and DataFrame writes)."""
    # Case 1: pd.read_* or pd.to_* function calls
    if isinstance(node.func, ast.Attribute):
        if hasattr(node.func.value, "id"):
            # Direct pandas import calls (pd.read_csv, etc.)
            if node.func.value.id == "pd":
                if not (len(node.args) > 0 and isinstance(node.args[0], ast.Str)):
                    return None

                filename = node.args[0].s
                method = node.func.attr

                is_read = method.startswith("read_")
                is_write = method.startswith("to_")

                if not (is_read or is_write):
                    return None

                return FileInfo(
                    filename=filename,
                    is_read=is_read,
                    is_write=is_write,
                    source_file=source_file,
                )

        # DataFrame method calls (df.to_csv, etc.)
        method = node.func.attr
        if method.startswith("to_"):
            if not (len(node.args) > 0 and isinstance(node.args[0], ast.Str)):
                return None

            filename = node.args[0].s
            return FileInfo(
                filename=filename, is_read=False, is_write=True, source_file=source_file
            )

    return None


def get_matplotlib_file_info(node: ast.Call, source_file: str) -> Optional[FileInfo]:
    """Extract file information from matplotlib save operations."""
    if not isinstance(node.func, ast.Attribute):
        return None

    # Check if it's a savefig call
    if node.func.attr != "savefig":
        return None

    # Handle both plt.savefig() and Figure.savefig()
    if hasattr(node.func.value, "id"):
        if node.func.value.id not in ["plt", "fig", "figure"]:
            return None

    # Get filename from first argument or fname keyword
    filename = None

    # Check positional argument
    if len(node.args) > 0 and isinstance(node.args[0], ast.Str):
        filename = node.args[0].s

    # Check for fname keyword argument
    for keyword in node.keywords:
        if keyword.arg == "fname" and isinstance(keyword.value, ast.Str):
            filename = keyword.value.s

    if not filename:
        return None

    return FileInfo(
        filename=filename, is_read=False, is_write=True, source_file=source_file
    )


class FileOperationFinder(ast.NodeVisitor):
    """AST visitor that finds file operations in Python code."""

    def __init__(self, source_file: str):
        self.source_file = source_file
        self.file_operations: List[FileInfo] = []

    def visit_Call(self, node: ast.Call):
        # Check for open() calls
        if file_info := get_open_file_info(node, self.source_file):
            self.file_operations.append(file_info)

        # Check for pandas operations
        if file_info := get_pandas_file_info(node, self.source_file):
            self.file_operations.append(file_info)

        # Check for matplotlib operations
        if file_info := get_matplotlib_file_info(node, self.source_file):
            self.file_operations.append(file_info)

        self.generic_visit(node)


class ModuleImportFinder(ast.NodeVisitor):
    """AST visitor that finds module imports in Python code."""

    def __init__(self, source_file: str, project_modules: Set[str]):
        self.source_file = source_file
        self.project_modules = project_modules
        self.imports: List[ModuleImport] = []

    def visit_Import(self, node: ast.Import):
        for name in node.names:
            base_module = name.name.split(".")[0]
            self.imports.append(
                ModuleImport(
                    module_name=name.name,
                    source_file=self.source_file,
                    is_from_import=False,
                    imported_names=[name.asname or name.name],
                    is_internal=base_module in self.project_modules,
                )
            )

    def visit_ImportFrom(self, node: ast.ImportFrom):
        if node.module:  # Ignore relative imports for simplicity
            base_module = node.module.split(".")[0]
            imported_names = [name.name for name in node.names]
            self.imports.append(
                ModuleImport(
                    module_name=node.module,
                    source_file=self.source_file,
                    is_from_import=True,
                    imported_names=imported_names,
                    is_internal=base_module in self.project_modules,
                )
            )


def get_project_modules(folder_path: str) -> Set[str]:
    """Find all potential internal module names in the project."""
    modules = set()
    for root, dirs, files in os.walk(folder_path):
        # Skip hidden directories (starting with .)
        dirs[:] = [d for d in dirs if not d.startswith('.')]
        
        for file in files:
            # Skip hidden files (starting with .)
            if file.startswith('.') or not file.endswith(".py"):
                continue
                
            # Get module name from file path
            rel_path = os.path.relpath(os.path.join(root, file), folder_path)
            module_name = os.path.splitext(rel_path)[0].replace(os.path.sep, ".")
            modules.add(module_name.split(".")[0])  # Add base module name
    return modules


def analyze_python_file(
    file_path: str, project_modules: Set[str]
) -> Tuple[List[FileInfo], List[ModuleImport]]:
    """Analyze a single Python file for file operations and imports."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            tree = ast.parse(f.read())

        # Find file operations
        file_finder = FileOperationFinder(file_path)
        file_finder.visit(tree)

        # Find imports
        import_finder = ModuleImportFinder(file_path, project_modules)
        import_finder.visit(tree)

        return file_finder.file_operations, import_finder.imports

    except (SyntaxError, UnicodeDecodeError, IOError) as e:
        print(f"Error processing {file_path}: {str(e)}")
        return [], []


def analyze_project(folder_path: str) -> ProjectModel:
    """
    Analyze a project folder and build a comprehensive project model.
    
    Args:
        folder_path: Path to the folder to analyze
        
    Returns:
        A ProjectModel containing the complete analysis results
    """
    model = ProjectModel(folder_path)
    project_modules = get_project_modules(folder_path)
    
    # Analyze all Python files in the project
    for root, dirs, files in os.walk(folder_path):
        # Skip hidden directories (starting with .)
        dirs[:] = [d for d in dirs if not d.startswith('.')]
        
        for file in files:
            # Skip hidden files (starting with .)
            if file.startswith('.') or not file.endswith(".py"):
                continue
                
            file_path = os.path.join(root, file)
            operations, imports = analyze_python_file(file_path, project_modules)
            
            # Add file operations to the model
            for op in operations:
                model.add_file_operation(op)
            
            # Add imports to the model
            rel_path = os.path.relpath(file_path, folder_path)
            for imp in imports:
                model.add_import(imp)
    
    # Build the graph representation
    model.build_graph()
    
    return model
