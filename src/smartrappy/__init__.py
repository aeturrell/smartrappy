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
from smartrappy.__main__ import analyse_and_visualise
from smartrappy.analyser import analyse_project
from smartrappy.models import (
    Edge,
    FileInfo,
    FileStatus,
    ModuleImport,
    Node,
    NodeType,
    ProjectModel,
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
    "analyse_and_visualise",
    # Models
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
