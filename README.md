# smartrappy

Smart reproducible analytical pipeline inspection.

![SVG logo of smartrappy](docs/logo.svg)

[![PyPI](https://img.shields.io/pypi/v/smartrappy.svg)](https://pypi.org/project/smartrappy/)
[![Status](https://img.shields.io/pypi/status/smartrappy.svg)](https://pypi.org/project/smartrappy/)
[![Python Version](https://img.shields.io/pypi/pyversions/smartrappy)](https://pypi.org/project/smartrappy)
[![License](https://img.shields.io/pypi/l/smartrappy)](https://opensource.org/licenses/MIT)
[![Read the documentation at https://aeturrell.github.io/smartrappy/](https://img.shields.io/badge/docs-passing-brightgreen)](https://aeturrell.github.io/smartrappy/)
[![Tests](https://github.com/aeturrell/smartrappy/workflows/Tests/badge.svg)](https://github.com/aeturrell/smartrappy/actions?workflow=Tests)
[![Codecov](https://codecov.io/gh/aeturrell/smartrappy/branch/main/graph/badge.svg)](https://codecov.io/gh/aeturrell/smartrappy)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Downloads](https://static.pepy.tech/badge/smartrappy)](https://pepy.tech/project/smartrappy)
[![Source](https://img.shields.io/badge/source%20code-github-lightgrey?style=for-the-badge)](https://github.com/aeturrell/smartrappy)

![Linux](https://img.shields.io/badge/Linux-FCC624?style=for-the-badge&logo=linux&logoColor=black)
![macOS](https://img.shields.io/badge/mac%20os-000000?style=for-the-badge&logo=macos&logoColor=F0F0F0)
![Windows](https://img.shields.io/badge/Windows-0078D6?style=for-the-badge&logo=windows&logoColor=white)



## Introduction

### What does this package do?

**smartrappy** analyses a Python project and infers the directed acyclic graph (DAG) of the code and data dependencies, including the last time any data were refreshed and whether the data exist at all on disk. It is not perfect, and will miss a lot in complex projects: but for simple projects using, say, `pd.read_csv()`, it does a good job of inferring the steps. The inferred DAG is then visualised, and there are several options for doing that—the default being to produce a visualisation in the terminal.

### What is **smartrappy** for?

**smartrappy** is designed to help you understand the dependencies in a project, especially in a context where there may be a lot of legacy code that resembles tangled spaghetti.

### Quickstart

To use **smartrappy** as a command-line tool:

```bash
smartrappy /path/to/your/project
```

Or to use it within a Python script:

```python
from smartrappy import analyse_project
from smartrappy.reporters import ConsoleReporter


model = analyse_project("/path/to/your/project")
reporter = ConsoleReporter()
reporter.generate_report(model)
```

### Installation

To install **smartrappy**, you can use `pip install smartrappy` or `uv add smartrappy` if you are using [Astral's uv](https://docs.astral.sh/uv/). You can also use it as a standalone command-line tool with uv and the `uvx` command:

```bash
uvx smartrappy path/to/your/project
```

### Documentation

You can find the full documentation for **smartrappy** at [https://aeturrell.github.io/smartrappy/](https://aeturrell.github.io/smartrappy/).
