"""
smartrappy
------------------------------------
Smart reproducible analytical pipeline execution
"""

import ast
import os
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version
from typing import Dict, List, NamedTuple, Set, Tuple

from graphviz import Digraph

try:
    __version__ = version("smartrappy")
except PackageNotFoundError:
    __version__ = "unknown"


class FileInfo(NamedTuple):
    """Information about a file operation found in Python code"""

    filename: str
    is_read: bool
    is_write: bool
    source_file: str


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


def get_open_file_info(node: ast.Call, source_file: str) -> FileInfo | None:
    """Extract file information from an open() function call"""
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


def get_pandas_file_info(node: ast.Call, source_file: str) -> FileInfo | None:
    """Extract file information from pandas operations (both pd.read_* and DataFrame writes)"""
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


def get_matplotlib_file_info(node: ast.Call, source_file: str) -> FileInfo | None:
    """Extract file information from matplotlib save operations"""
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
    """AST visitor that finds file operations in Python code"""

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


def analyze_python_file(file_path: str) -> List[FileInfo]:
    """Analyze a single Python file for file operations"""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            tree = ast.parse(f.read())

        finder = FileOperationFinder(file_path)
        finder.visit(tree)
        return finder.file_operations

    except (SyntaxError, UnicodeDecodeError, IOError) as e:
        print(f"Error processing {file_path}: {str(e)}")
        return []


def find_file_operations(folder_path: str) -> Dict[str, List[FileInfo]]:
    """
    Find all file operations in Python files within a folder.

    Args:
        folder_path: Path to the folder to analyze

    Returns:
        Dictionary mapping filenames to lists of FileInfo objects
    """
    all_operations: Dict[str, List[FileInfo]] = {}

    # Walk through all Python files in the folder
    for root, _, files in os.walk(folder_path):
        for file in files:
            if not file.endswith(".py"):
                continue

            file_path = os.path.join(root, file)
            operations = analyze_python_file(file_path)

            # Group operations by filename
            for op in operations:
                if op.filename not in all_operations:
                    all_operations[op.filename] = []
                all_operations[op.filename].append(op)

    return all_operations


def print_analysis(operations: Dict[str, List[FileInfo]]):
    """Print a human-readable analysis of file operations"""
    if not operations:
        print("No file operations found.")
        return

    print("\nFile Operations Analysis:")
    print("=" * 80)

    for filename, file_ops in sorted(operations.items()):
        print(f"\nFile: {filename}")

        # Determine overall operation type
        has_read = any(op.is_read for op in file_ops)
        has_write = any(op.is_write for op in file_ops)

        if has_read and has_write:
            op_type = "READ/WRITE"
        elif has_read:
            op_type = "READ"
        else:
            op_type = "WRITE"

        print(f"Operation: {op_type}")

        print("Referenced in:")
        sources = sorted(set(op.source_file for op in file_ops))
        for source in sources:
            print(f"  - {source}")


class FileStatus(NamedTuple):
    """Information about a file's status on disk"""

    exists: bool
    last_modified: datetime | None


def get_file_status(filepath: str) -> FileStatus:
    """
    Get file existence and modification time information.

    Args:
        filepath: Path to the file to check

    Returns:
        FileStatus object with existence and modification time
    """
    if os.path.exists(filepath):
        mtime = datetime.fromtimestamp(os.path.getmtime(filepath))
        return FileStatus(exists=True, last_modified=mtime)
    return FileStatus(exists=False, last_modified=None)


def generate_mermaid_visualization(
    operations: Dict[str, List[FileInfo]], base_path: str
) -> str:
    """
    Generate a Mermaid graph visualization of file operations with file status information.

    Args:
        operations: Dictionary mapping filenames to lists of FileInfo objects
        base_path: Base path for resolving relative file paths

    Returns:
        String containing Mermaid graph definition
    """
    # Track all unique scripts and files
    scripts = set()
    data_files = set()
    relationships = set()
    file_statuses = {}

    # Collect all nodes and relationships
    for filename, file_ops in operations.items():
        data_files.add(filename)

        # Get file status
        filepath = os.path.join(base_path, filename)
        file_statuses[filename] = get_file_status(filepath)

        for op in file_ops:
            script_name = os.path.basename(op.source_file)
            scripts.add(script_name)

            if op.is_read:
                relationships.add((filename, script_name))
            if op.is_write:
                relationships.add((script_name, filename))

    # Generate Mermaid markup
    mermaid = [
        "graph TD",
        "    %% Style definitions",
        "    classDef scriptNode fill:#90EE90,stroke:#333,stroke-width:2px;",
        "    classDef fileNode fill:#FFB6C1,stroke:#333,stroke-width:2px;",
        "    classDef missingFile fill:#FFB6C1,stroke:#FF0000,stroke-width:3px,stroke-dasharray: 5 5;",
        "",
        "    %% Nodes",
    ]

    # Add script nodes
    for script in sorted(scripts):
        node_id = f"script_{hash(script) & 0xFFFFFF}"
        mermaid.append(f'    {node_id}["{script}"]:::scriptNode')

    # Add file nodes with status information
    for file in sorted(data_files):
        node_id = f"file_{hash(file) & 0xFFFFFF}"
        status = file_statuses[file]

        if status.exists:
            mod_time = status.last_modified.strftime("%Y-%m-%d %H:%M:%S")
            label = f"{file}<br/><small>Modified: {mod_time}</small>"
            mermaid.append(f'    {node_id}["{label}"]:::fileNode')
        else:
            label = f"{file}<br/><small>File does not exist</small>"
            mermaid.append(f'    {node_id}["{label}"]:::missingFile')

    mermaid.append("")
    mermaid.append("    %% Relationships")

    # Add relationships
    for src, dst in sorted(relationships):
        src_id = (
            f"script_{hash(src) & 0xFFFFFF}"
            if src in scripts
            else f"file_{hash(src) & 0xFFFFFF}"
        )
        dst_id = (
            f"script_{hash(dst) & 0xFFFFFF}"
            if dst in scripts
            else f"file_{hash(dst) & 0xFFFFFF}"
        )
        mermaid.append(f"    {src_id} --> {dst_id}")

    return "\n".join(mermaid)


# def analyze_and_visualize(folder_path: str):
#     """
#     Analyze Python files in a folder and create a visualization of file operations.

#     Args:
#         folder_path: Path to the folder to analyze
#     """
#     operations = find_file_operations(folder_path)
#     print_analysis(operations)

#     print("\nGenerating visualization...")
#     mermaid = generate_mermaid_visualization(operations, folder_path)
#     print("\nMermaid Graph Definition:")
#     print(mermaid)


#  Convert mermaid diagram to CSV
#  https://github.com/mermaid-js/mermaid-cli?tab=readme-ov-file#usage


def generate_graphviz_visualization(
    operations: Dict[str, List[FileInfo]], base_path: str
) -> Digraph:
    """
    Generate a Graphviz visualization of file operations with file status information.

    Args:
        operations: Dictionary mapping filenames to lists of FileInfo objects
        base_path: Base path for resolving relative file paths

    Returns:
        Graphviz Digraph object
    """
    # Create a new directed graph
    dot = Digraph(comment="File Operations Graph")
    dot.attr(rankdir="TB")  # Top to bottom layout

    # Track all unique scripts and files
    scripts = set()
    data_files = set()
    relationships = set()
    file_statuses = {}

    # Collect all nodes and relationships
    for filename, file_ops in operations.items():
        data_files.add(filename)

        # Get file status
        filepath = os.path.join(base_path, filename)
        file_statuses[filename] = get_file_status(filepath)

        for op in file_ops:
            script_name = os.path.basename(op.source_file)
            scripts.add(script_name)

            if op.is_read:
                relationships.add((filename, script_name))
            if op.is_write:
                relationships.add((script_name, filename))

    # Define node styles
    dot.attr("node", shape="box", style="filled")

    # Add script nodes
    for script in sorted(scripts):
        node_id = f"script_{hash(script) & 0xFFFFFF}"
        dot.node(
            node_id,
            script,
            fillcolor="#90EE90",  # Light green
            color="#333333",
            penwidth="2.0",
        )

    # Add file nodes with status information
    for file in sorted(data_files):
        node_id = f"file_{hash(file) & 0xFFFFFF}"
        status = file_statuses[file]

        if status.exists:
            mod_time = status.last_modified.strftime("%Y-%m-%d %H:%M:%S")
            label = f"{file}\nModified: {mod_time}"
            dot.node(
                node_id,
                label,
                fillcolor="#FFB6C1",  # Light pink
                color="#333333",
                penwidth="2.0",
            )
        else:
            label = f"{file}\nFile does not exist"
            dot.node(
                node_id,
                label,
                fillcolor="#FFB6C1",  # Light pink
                color="#FF0000",  # Red border
                penwidth="3.0",
                style="filled,dashed",
            )

    # Add relationships
    dot.attr("edge", color="#333333")
    for src, dst in sorted(relationships):
        src_id = (
            f"script_{hash(src) & 0xFFFFFF}"
            if src in scripts
            else f"file_{hash(src) & 0xFFFFFF}"
        )
        dst_id = (
            f"script_{hash(dst) & 0xFFFFFF}"
            if dst in scripts
            else f"file_{hash(dst) & 0xFFFFFF}"
        )
        dot.edge(src_id, dst_id)

    return dot


def analyze_and_visualize(folder_path: str, output_path: str = "file_operations"):
    """
    Analyze Python files in a folder and create a visualization of file operations.

    Args:
        folder_path: Path to the folder to analyze
        output_path: Base name for the output files (without extension)
    """
    operations = find_file_operations(folder_path)
    print_analysis(operations)

    print("\nGenerating visualization...")
    dot = generate_graphviz_visualization(operations, folder_path)

    # Save the graph in multiple formats
    dot.render(
        output_path, view=True, format="pdf"
    )  # Creates both .pdf and .pdf.dot files
    print(f"\nVisualization saved as {output_path}.pdf")


class ModuleImport(NamedTuple):
    """Information about a module import found in Python code"""

    module_name: str
    source_file: str
    is_from_import: bool
    imported_names: List[str]


class FileInfo(NamedTuple):
    """Information about a file operation found in Python code"""

    filename: str
    is_read: bool
    is_write: bool
    source_file: str


class ModuleImportFinder(ast.NodeVisitor):
    """AST visitor that finds module imports in Python code"""

    def __init__(self, source_file: str):
        self.source_file = source_file
        self.imports: List[ModuleImport] = []

    def visit_Import(self, node: ast.Import):
        for name in node.names:
            self.imports.append(
                ModuleImport(
                    module_name=name.name,
                    source_file=self.source_file,
                    is_from_import=False,
                    imported_names=[name.asname or name.name],
                )
            )

    def visit_ImportFrom(self, node: ast.ImportFrom):
        if node.module:  # Ignore relative imports for simplicity
            imported_names = [name.name for name in node.names]
            self.imports.append(
                ModuleImport(
                    module_name=node.module,
                    source_file=self.source_file,
                    is_from_import=True,
                    imported_names=imported_names,
                )
            )


def analyze_python_file_with_imports(
    file_path: str,
) -> Tuple[List[FileInfo], List[ModuleImport]]:
    """Analyze a single Python file for file operations and imports"""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            tree = ast.parse(f.read())

        # Find file operations
        file_finder = FileOperationFinder(file_path)
        file_finder.visit(tree)

        # Find imports
        import_finder = ModuleImportFinder(file_path)
        import_finder.visit(tree)

        return file_finder.file_operations, import_finder.imports

    except (SyntaxError, UnicodeDecodeError, IOError) as e:
        print(f"Error processing {file_path}: {str(e)}")
        return [], []


def find_all_operations(
    folder_path: str,
) -> Tuple[Dict[str, List[FileInfo]], Dict[str, List[ModuleImport]]]:
    """Find all file operations and imports in Python files within a folder"""
    all_file_ops: Dict[str, List[FileInfo]] = {}
    all_imports: Dict[str, List[ModuleImport]] = {}

    for root, _, files in os.walk(folder_path):
        for file in files:
            if not file.endswith(".py"):
                continue

            file_path = os.path.join(root, file)
            operations, imports = analyze_python_file_with_imports(file_path)

            # Group file operations
            for op in operations:
                if op.filename not in all_file_ops:
                    all_file_ops[op.filename] = []
                all_file_ops[op.filename].append(op)

            # Group imports by source file
            rel_path = os.path.relpath(file_path, folder_path)
            all_imports[rel_path] = imports

    return all_file_ops, all_imports


def generate_enhanced_graphviz(
    operations: Dict[str, List[FileInfo]],
    imports: Dict[str, List[ModuleImport]],
    base_path: str,
) -> Digraph:
    """Generate a Graphviz visualization including both file operations and imports"""
    dot = Digraph(comment="Enhanced File Operations and Import Graph")
    dot.attr(rankdir="TB")

    # Track nodes and relationships
    scripts = set()
    data_files = set()
    module_nodes = set()
    relationships = set()
    file_statuses = {}

    # Process file operations
    for filename, file_ops in operations.items():
        data_files.add(filename)
        filepath = os.path.join(base_path, filename)
        file_statuses[filename] = get_file_status(filepath)

        for op in file_ops:
            script_name = os.path.basename(op.source_file)
            scripts.add(script_name)
            if op.is_read:
                relationships.add((filename, script_name))
            if op.is_write:
                relationships.add((script_name, filename))

    # Process imports
    for script_path, script_imports in imports.items():
        script_name = os.path.basename(script_path)
        scripts.add(script_name)

        for imp in script_imports:
            module_nodes.add(imp.module_name)
            relationships.add((imp.module_name, script_name))

    # Define node styles
    dot.attr("node", shape="box", style="filled")

    # Add module nodes
    for module in sorted(module_nodes):
        node_id = f"module_{hash(module) & 0xFFFFFF}"
        dot.node(
            node_id,
            module,
            fillcolor="#ADD8E6",  # Light blue
            color="#333333",
            penwidth="2.0",
        )

    # Add script nodes
    for script in sorted(scripts):
        node_id = f"script_{hash(script) & 0xFFFFFF}"
        dot.node(
            node_id,
            script,
            fillcolor="#90EE90",  # Light green
            color="#333333",
            penwidth="2.0",
        )

    # Add file nodes
    for file in sorted(data_files):
        node_id = f"file_{hash(file) & 0xFFFFFF}"
        status = file_statuses[file]

        if status.exists:
            mod_time = status.last_modified.strftime("%Y-%m-%d %H:%M:%S")
            label = f"{file}\nModified: {mod_time}"
            dot.node(
                node_id,
                label,
                fillcolor="#FFB6C1",  # Light pink
                color="#333333",
                penwidth="2.0",
            )
        else:
            label = f"{file}\nFile does not exist"
            dot.node(
                node_id,
                label,
                fillcolor="#FFB6C1",
                color="#FF0000",
                penwidth="3.0",
                style="filled,dashed",
            )

    # Add relationships
    dot.attr("edge", color="#333333")
    for src, dst in sorted(relationships):
        src_id = get_node_id(src, scripts, module_nodes)
        dst_id = get_node_id(dst, scripts, module_nodes)
        dot.edge(src_id, dst_id)

    return dot


def get_node_id(name: str, scripts: Set[str], modules: Set[str]) -> str:
    """Helper function to get the correct node ID based on node type"""
    if name in scripts:
        return f"script_{hash(name) & 0xFFFFFF}"
    elif name in modules:
        return f"module_{hash(name) & 0xFFFFFF}"
    else:
        return f"file_{hash(name) & 0xFFFFFF}"


def analyze_and_visualize(folder_path: str, output_path: str = "file_operations"):
    """Analyze Python files and create an enhanced visualization"""
    operations, imports = find_all_operations(folder_path)

    print("\nFile Operations and Import Analysis:")
    print("=" * 80)

    # Print file operations
    for filename, file_ops in sorted(operations.items()):
        print(f"\nFile: {filename}")
        has_read = any(op.is_read for op in file_ops)
        has_write = any(op.is_write for op in file_ops)
        op_type = (
            "READ/WRITE"
            if has_read and has_write
            else ("READ" if has_read else "WRITE")
        )
        print(f"Operation: {op_type}")
        print("Referenced in:")
        sources = sorted(set(op.source_file for op in file_ops))
        for source in sources:
            print(f"  - {source}")

    # Print import analysis
    print("\nModule Imports:")
    for script, script_imports in sorted(imports.items()):
        if script_imports:
            print(f"\nScript: {script}")
            for imp in script_imports:
                names = ", ".join(imp.imported_names)
                import_type = "from" if imp.is_from_import else "import"
                print(f"  - {import_type} {imp.module_name} ({names})")

    print("\nGenerating visualization...")
    dot = generate_enhanced_graphviz(operations, imports, folder_path)
    dot.render(output_path, view=True, format="pdf")
    print(f"\nVisualization saved as {output_path}.pdf")
