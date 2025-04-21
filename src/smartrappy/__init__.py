"""
smartrappy
------------------------------------
Smart reproducible analytical pipeline execution
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("smartrappy")
except PackageNotFoundError:
    __version__ = "unknown"

# Import core components first
# Import CLI functions last to avoid circular imports
from smartrappy.analyser import analyse_project
from smartrappy.models import (
    DatabaseInfo,
    Edge,
    FileInfo,
    FileStatus,
    ModuleImport,
    Node,
    NodeType,
    ProjectModel,
)
from smartrappy.qmd_parser import (
    analyse_qmd_file,
    extract_markdown_resources,
    extract_python_chunks,
)
from smartrappy.reporters import (
    ConsoleReporter,
    GraphvizReporter,
    JsonReporter,
    MermaidReporter,
    Reporter,
    get_reporter,
)

__all__ = [
    # Main functions
    "analyse_project",
    "analyse_qmd_file",
    "extract_python_chunks",
    "extract_markdown_resources",  # New export
    # Models
    "DatabaseInfo",
    "Edge",
    "FileInfo",
    "FileStatus",
    "ModuleImport",
    "Node",
    "NodeType",
    "ProjectModel",
    # Reporters
    "Reporter",
    "ConsoleReporter",
    "GraphvizReporter",
    "MermaidReporter",
    "JsonReporter",
    "get_reporter",
]
