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

