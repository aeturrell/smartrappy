"""Data models for smartrappy."""

import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional


class FileInfo(NamedTuple):
    """Information about a file operation found in Python code."""

    filename: str
    is_read: bool
    is_write: bool
    source_file: str


class FileStatus(NamedTuple):
    """Information about a file's status on disk."""

    exists: bool
    last_modified: Optional[datetime] = None


class ModuleImport(NamedTuple):
    """Information about a module import found in Python code."""

    module_name: str
    source_file: str
    is_from_import: bool
    imported_names: List[str]
    is_internal: bool


class DatabaseInfo(NamedTuple):
    """Information about a database operation found in Python code."""

    db_name: str  # Name or identifier of the database
    connection_string: Optional[str]  # Connection string (if available)
    db_type: str  # Type of database (e.g., "postgresql", "mysql", "sqlite")
    is_read: bool  # Whether data is read from the database
    is_write: bool  # Whether data is written to the database
    source_file: str  # File containing the database operation
    conn_var_name: Optional[str] = None  # Connection variable name if applicable
    uses_conn_var: Optional[str] = None  # If this operation uses a connection variable


class NodeType:
    """Enumeration of node types in the project graph."""

    SCRIPT = "script"
    DATA_FILE = "data_file"
    EXTERNAL_MODULE = "external_module"
    INTERNAL_MODULE = "internal_module"
    DATABASE = "database"
    QUARTO_DOCUMENT = "quarto_document"  # New node type for Quarto documents


class Node(NamedTuple):
    """A node in the project dependency graph."""

    id: str
    name: str
    type: str
    metadata: dict


class Edge(NamedTuple):
    """An edge in the project dependency graph."""

    source: str
    target: str
    type: str


class ProjectModel:
    """A complete model of the project's structure and dependencies."""

    def __init__(self, base_path: str, internal_only: bool = False):
        self.base_path = Path(base_path)
        self.internal_only = internal_only
        self.nodes: Dict[str, Node] = {}
        self.edges: List[Edge] = []
        self.file_operations: Dict[str, List[FileInfo]] = {}
        self.imports: Dict[str, List[ModuleImport]] = {}
        self.file_statuses: Dict[str, FileStatus] = {}
        self.database_operations: Dict[str, List[DatabaseInfo]] = {}

    def get_node_id(self, name: str, node_type: str) -> str:
        """Generate a consistent node ID based on name and type."""
        return f"{node_type}_{hash(name) & 0xFFFFFF}"

    def add_node(
        self, name: str, node_type: str, metadata: Optional[dict] = None
    ) -> str:
        """Add a node to the model and return its ID."""
        metadata = metadata or {}
        node_id = self.get_node_id(name, node_type)

        if node_id not in self.nodes:
            self.nodes[node_id] = Node(
                id=node_id, name=name, type=node_type, metadata=metadata
            )

        return node_id

    def add_edge(
        self, source_id: str, target_id: str, edge_type: str = "dependency"
    ) -> None:
        """Add an edge between two nodes."""
        # Prevent duplicate edges
        for edge in self.edges:
            if (
                edge.source == source_id
                and edge.target == target_id
                and edge.type == edge_type
            ):
                return

        self.edges.append(Edge(source=source_id, target=target_id, type=edge_type))

    def add_file_operation(self, operation: FileInfo) -> None:
        """Add a file operation to the model."""
        if operation.filename not in self.file_operations:
            self.file_operations[operation.filename] = []

        # Prevent duplicate operations
        for op in self.file_operations[operation.filename]:
            if (
                op.source_file == operation.source_file
                and op.is_read == operation.is_read
                and op.is_write == operation.is_write
            ):
                return

        self.file_operations[operation.filename].append(operation)

        # Update file status if not already stored
        if operation.filename not in self.file_statuses:
            filepath = self.base_path / operation.filename
            self.file_statuses[operation.filename] = get_file_status(filepath)

    def add_import(self, import_info: ModuleImport) -> None:
        """Add a module import to the model."""
        if import_info.source_file not in self.imports:
            self.imports[import_info.source_file] = []

        # Prevent duplicate imports
        for imp in self.imports[import_info.source_file]:
            if (
                imp.module_name == import_info.module_name
                and imp.is_from_import == import_info.is_from_import
            ):
                return

        self.imports[import_info.source_file].append(import_info)

    def add_database_operation(self, operation: DatabaseInfo) -> None:
        """Add a database operation to the model, handling connection variables."""
        db_name_to_use = operation.db_name

        # If this operation uses a connection variable, prioritize its database name
        if hasattr(operation, "uses_conn_var") and operation.uses_conn_var:
            conn_var = operation.uses_conn_var
            # Look through existing operations to find the referenced connection
            for existing_ops in self.database_operations.values():
                for op in existing_ops:
                    if hasattr(op, "conn_var_name") and op.conn_var_name == conn_var:
                        db_name_to_use = op.db_name
                        break

        # Now add the operation under the appropriate database name
        if db_name_to_use not in self.database_operations:
            self.database_operations[db_name_to_use] = []

        # Prevent duplicate operations
        for op in self.database_operations[db_name_to_use]:
            if (
                op.source_file == operation.source_file
                and op.is_read == operation.is_read
                and op.is_write == operation.is_write
                and op.db_type == operation.db_type
            ):
                return

        self.database_operations[db_name_to_use].append(operation)

    def build_graph(self) -> None:
        """Build the graph representation from file operations, database operations, and imports."""
        # Process file operations
        for filename, operations in self.file_operations.items():
            file_node_id = self.add_node(
                filename,
                NodeType.DATA_FILE,
                {"status": self.file_statuses.get(filename, FileStatus(exists=False))},
            )

            for op in operations:
                script_name = os.path.basename(op.source_file)

                # Check if the file is a QMD file or a Python script
                node_type = (
                    NodeType.QUARTO_DOCUMENT
                    if script_name.endswith(".qmd")
                    else NodeType.SCRIPT
                )
                script_node_id = self.add_node(script_name, node_type)

                if op.is_read:
                    self.add_edge(file_node_id, script_node_id, "read")
                if op.is_write:
                    self.add_edge(script_node_id, file_node_id, "write")

        # Process database operations
        for db_name, operations in self.database_operations.items():
            db_node_id = self.add_node(
                db_name,
                NodeType.DATABASE,
                {"db_type": operations[0].db_type},  # Use type from first operation
            )

            for op in operations:
                script_name = os.path.basename(op.source_file)
                # Check if the file is a QMD file or a Python script
                node_type = (
                    NodeType.QUARTO_DOCUMENT
                    if script_name.endswith(".qmd")
                    else NodeType.SCRIPT
                )
                script_node_id = self.add_node(script_name, node_type)

                if op.is_read:
                    self.add_edge(db_node_id, script_node_id, "read")
                if op.is_write:
                    self.add_edge(script_node_id, db_node_id, "write")

        # Process imports - create more detailed nodes
        for source_file, imports in self.imports.items():
            script_name = os.path.basename(source_file)
            # Check if the file is a QMD file or a Python script
            node_type = (
                NodeType.QUARTO_DOCUMENT
                if script_name.endswith(".qmd")
                else NodeType.SCRIPT
            )
            script_node_id = self.add_node(script_name, node_type)

            for imp in imports:
                # Skip external modules if internal_only is True
                if self.internal_only and not imp.is_internal:
                    continue

                # Get base module name without path
                base_module_name = os.path.basename(imp.module_name.replace(".", "/"))
                module_display_name = base_module_name

                # Create separate nodes for each imported item if it's a from-import
                if imp.is_from_import and imp.imported_names:
                    for imported_name in imp.imported_names:
                        # Create detailed import name with module:function format
                        detailed_name = f"{module_display_name}:{imported_name}"
                        node_type = (
                            NodeType.INTERNAL_MODULE
                            if imp.is_internal
                            else NodeType.EXTERNAL_MODULE
                        )

                        import_node_id = self.add_node(
                            detailed_name,
                            node_type,
                            {
                                "module": module_display_name,
                                "imported_name": imported_name,
                                "is_from_import": True,
                            },
                        )
                        self.add_edge(import_node_id, script_node_id, "import")
                else:
                    # For regular imports, just use the module name
                    node_type = (
                        NodeType.INTERNAL_MODULE
                        if imp.is_internal
                        else NodeType.EXTERNAL_MODULE
                    )
                    import_node_id = self.add_node(module_display_name, node_type)
                    self.add_edge(import_node_id, script_node_id, "import")


def get_file_status(filepath: Path) -> FileStatus:
    """Get file existence and modification time information."""
    if filepath.exists():
        mtime = datetime.fromtimestamp(filepath.stat().st_mtime)
        return FileStatus(exists=True, last_modified=mtime)
    return FileStatus(exists=False)
