"""Data models for smartrappy."""

import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Set


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


class NodeType:
    """Enumeration of node types in the project graph."""
    
    SCRIPT = "script"
    DATA_FILE = "data_file"
    EXTERNAL_MODULE = "external_module"
    INTERNAL_MODULE = "internal_module"


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
    
    def __init__(self, base_path: str):
        self.base_path = Path(base_path)
        self.nodes: Dict[str, Node] = {}
        self.edges: List[Edge] = []
        self.file_operations: Dict[str, List[FileInfo]] = {}
        self.imports: Dict[str, List[ModuleImport]] = {}
        self.file_statuses: Dict[str, FileStatus] = {}
    
    def get_node_id(self, name: str, node_type: str) -> str:
        """Generate a consistent node ID based on name and type."""
        return f"{node_type}_{hash(name) & 0xFFFFFF}"
    
    def add_node(self, name: str, node_type: str, metadata: Optional[dict] = None) -> str:
        """Add a node to the model and return its ID."""
        metadata = metadata or {}
        node_id = self.get_node_id(name, node_type)
        
        if node_id not in self.nodes:
            self.nodes[node_id] = Node(
                id=node_id,
                name=name,
                type=node_type,
                metadata=metadata
            )
        
        return node_id
    
    def add_edge(self, source_id: str, target_id: str, edge_type: str = "dependency") -> None:
        """Add an edge between two nodes."""
        # Prevent duplicate edges
        for edge in self.edges:
            if edge.source == source_id and edge.target == target_id and edge.type == edge_type:
                return
                
        self.edges.append(Edge(source=source_id, target=target_id, type=edge_type))
    
    def add_file_operation(self, operation: FileInfo) -> None:
        """Add a file operation to the model."""
        if operation.filename not in self.file_operations:
            self.file_operations[operation.filename] = []
        
        # Prevent duplicate operations
        for op in self.file_operations[operation.filename]:
            if (op.source_file == operation.source_file and 
                op.is_read == operation.is_read and 
                op.is_write == operation.is_write):
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
            if (imp.module_name == import_info.module_name and 
                imp.is_from_import == import_info.is_from_import):
                return
                
        self.imports[import_info.source_file].append(import_info)
    
    def build_graph(self) -> None:
        """Build the graph representation from file operations and imports."""
        # Process file operations
        for filename, operations in self.file_operations.items():
            file_node_id = self.add_node(filename, NodeType.DATA_FILE, {
                "status": self.file_statuses.get(filename, FileStatus(exists=False))
            })
            
            for op in operations:
                script_name = os.path.basename(op.source_file)
                script_node_id = self.add_node(script_name, NodeType.SCRIPT)
                
                if op.is_read:
                    self.add_edge(file_node_id, script_node_id, "read")
                if op.is_write:
                    self.add_edge(script_node_id, file_node_id, "write")
        
        # Process imports
        for source_file, imports in self.imports.items():
            script_name = os.path.basename(source_file)
            script_node_id = self.add_node(script_name, NodeType.SCRIPT)
            
            for imp in imports:
                node_type = NodeType.INTERNAL_MODULE if imp.is_internal else NodeType.EXTERNAL_MODULE
                module_node_id = self.add_node(imp.module_name, node_type)
                self.add_edge(module_node_id, script_node_id, "import")


def get_file_status(filepath: Path) -> FileStatus:
    """Get file existence and modification time information."""
    if filepath.exists():
        mtime = datetime.fromtimestamp(filepath.stat().st_mtime)
        return FileStatus(exists=True, last_modified=mtime)
    return FileStatus(exists=False)
