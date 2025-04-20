"""
smartrappy
------------------------------------
Smart reproducible analytical pipeline visualisation.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("smartrappy")
except PackageNotFoundError:
    __version__ = "unknown"

# Import main functionality for ease of use
from smartrappy.__main__ import analyze_and_visualize
from smartrappy.analyzer import analyze_project
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
    "analyze_project",
    "analyze_and_visualize",
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
