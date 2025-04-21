"""Reporters for smartrappy analysis results."""

import json
import os
from abc import ABC, abstractmethod
from typing import Dict, Optional, Set

from graphviz import Digraph
from rich.console import Console
from rich.text import Text
from rich.tree import Tree

from smartrappy.models import NodeType, ProjectModel


class Reporter(ABC):
    """Base class for all reporters."""

    @abstractmethod
    def generate_report(
        self, model: ProjectModel, output_path: Optional[str] = None
    ) -> None:
        """Generate a report from the project model."""
        pass


class ConsoleReporter(Reporter):
    """Report analysis results to the console."""

    def generate_report(
        self, model: ProjectModel, output_path: Optional[str] = None
    ) -> None:
        """Generate a console report from the project model."""
        console = Console()

        # Print header
        console.print(
            "\n[bold cyan]File Operations, Database Operations, and Import Analysis[/bold cyan]"
        )
        console.print("=" * 80)

        # Print file operations
        for filename, file_ops in sorted(model.file_operations.items()):
            console.print(f"\n[bold]File:[/bold] {filename}")
            has_read = any(op.is_read for op in file_ops)
            has_write = any(op.is_write for op in file_ops)
            op_type = (
                "READ/WRITE"
                if has_read and has_write
                else ("READ" if has_read else "WRITE")
            )
            console.print(f"[bold]Operation:[/bold] {op_type}")
            console.print("[bold]Referenced in:[/bold]")
            sources = sorted(set(op.source_file for op in file_ops))
            for source in sources:
                console.print(f"  - {source}")

        if model.database_operations:
            console.print("\n[bold purple]💽 Database Operations:[/bold purple]")
            for db_name, db_ops in sorted(model.database_operations.items()):
                console.print(f"\n[bold]Database:[/bold] {db_name}")
                db_type = db_ops[0].db_type  # Get type from first operation
                console.print(f"[bold]Type:[/bold] {db_type}")

                has_read = any(op.is_read for op in db_ops)
                has_write = any(op.is_write for op in db_ops)
                op_type = (
                    "READ/WRITE"
                    if has_read and has_write
                    else ("READ" if has_read else "WRITE")
                )
                console.print(f"[bold]Operation:[/bold] {op_type}")

                console.print("[bold]Referenced in:[/bold]")
                sources = sorted(set(op.source_file for op in db_ops))
                for source in sources:
                    console.print(f"  - {source}")

        # Print import analysis
        console.print("\n[bold]Module Imports:[/bold]")
        for script, script_imports in sorted(model.imports.items()):
            if script_imports:
                script_name = os.path.basename(script)
                console.print(f"\n[bold]Script:[/bold] {script_name}")
                for imp in script_imports:
                    # Get module display name with .py extension for Python modules
                    module_display = os.path.basename(imp.module_name.replace(".", "/"))
                    # if not module_display.endswith(".py") and "." not in module_display:
                    #     module_display = f"{module_display}.py"

                    import_type = "from" if imp.is_from_import else "import"
                    module_type = (
                        "[blue]internal[/blue]"
                        if imp.is_internal
                        else "[red]external[/red]"
                    )

                    # For 'from' imports, show as module:imported_names
                    if imp.is_from_import:
                        detailed_imports = [
                            f"{module_display}:{name}" for name in imp.imported_names
                        ]
                        detailed_str = ", ".join(detailed_imports)
                        console.print(
                            f"  - {import_type} {imp.module_name} → {detailed_str} [{module_type}]"
                        )
                    else:
                        console.print(
                            f"  - {import_type} {module_display} [{module_type}]"
                        )

        # Create and display terminal visualisation
        console.print("\n[bold cyan]Terminal Visualisation[/bold cyan]")
        tree = self._create_terminal_tree(model)
        console.print(tree)

    def _create_terminal_tree(self, model: ProjectModel) -> Tree:
        """Create a rich Tree visualisation of the dependency graph."""
        # Create the main tree
        tree = Tree("📦 Project Dependencies", guide_style="bold cyan")

        # Track all nodes and their dependencies
        dependencies: Dict[str, Set[str]] = {}  # node_id -> set of dependency node_ids

        # Process edges to build dependency map
        for edge in model.edges:
            if edge.target not in dependencies:
                dependencies[edge.target] = set()
            dependencies[edge.target].add(edge.source)

        # Find root nodes (nodes with no incoming edges)
        all_nodes = set(model.nodes.keys())
        dependency_targets = set()
        for deps in dependencies.values():
            dependency_targets.update(deps)
        root_nodes = all_nodes - dependency_targets

        # Helper function to get node style
        def get_node_style(node_type: str, name: str) -> Text:
            icons = {
                NodeType.SCRIPT: "📜",
                NodeType.EXTERNAL_MODULE: "📦",
                NodeType.INTERNAL_MODULE: "🔧",
                NodeType.DATA_FILE: "📄",
                NodeType.DATABASE: "💽",  # Using the database symbol for database nodes
                NodeType.QUARTO_DOCUMENT: "📰",  # New icon for Quarto documents
            }
            colors = {
                NodeType.SCRIPT: "green",
                NodeType.EXTERNAL_MODULE: "red",
                NodeType.INTERNAL_MODULE: "blue",
                NodeType.DATA_FILE: "magenta",
                NodeType.DATABASE: "purple",  # Using purple color for database nodes
                NodeType.QUARTO_DOCUMENT: "cyan",  # Use cyan color for Quarto documents
            }
            return Text(
                f"{icons.get(node_type, '❓')} {name}",
                style=colors.get(node_type, "white"),
            )

        # Helper function to recursively build tree
        def build_tree(node_id: str, seen: Set[str], parent_tree: Tree) -> None:
            if node_id in seen:
                return

            node = model.nodes[node_id]
            seen.add(node_id)

            # Add node to tree
            node_tree = parent_tree.add(get_node_style(node.type, node.name))

            # For database nodes, add type information
            if node.type == NodeType.DATABASE and "db_type" in node.metadata:
                node_tree.add(Text(f"Type: {node.metadata['db_type']}", "purple"))

            # Add dependencies
            for dep_id in sorted(dependencies.get(node_id, set())):
                if dep_id not in seen:
                    build_tree(dep_id, seen.copy(), node_tree)
                else:
                    # Show circular dependency
                    dep_node = model.nodes[dep_id]
                    node_tree.add(Text(f"↻ {dep_node.name} (circular)", "yellow"))

        # Build tree from each root node
        for root_id in sorted(root_nodes):
            build_tree(root_id, set(), tree)

        return tree


class GraphvizReporter(Reporter):
    """Generate a Graphviz visualisation of the project model. Exports as PDF"""

    def generate_report(
        self, model: ProjectModel, output_path: Optional[str] = None
    ) -> None:
        """Generate a Graphviz visualisation from the project model."""
        if not output_path:
            output_path = "project_graph"

        # Create a new directed graph
        dot = Digraph(comment="Project Dependency Graph")
        dot.attr(rankdir="TB")  # Top to bottom layout

        # Define node styles
        dot.attr("node", shape="box", style="filled")

        # Add nodes
        for node_id, node in model.nodes.items():
            if node.type == NodeType.SCRIPT:
                dot.node(
                    node_id,
                    node.name,
                    fillcolor="#90EE90",  # Light green
                    color="#333333",
                    penwidth="2.0",
                )
            elif node.type == NodeType.DATA_FILE:
                # Handle file status for data files
                if "status" in node.metadata:
                    status = node.metadata["status"]
                    if status.exists:
                        mod_time = status.last_modified.strftime("%Y-%m-%d %H:%M:%S")
                        label = f"{node.name}\nModified: {mod_time}"
                        dot.node(
                            node_id,
                            label,
                            fillcolor="#FFB6C1",  # Light pink
                            color="#333333",
                            penwidth="2.0",
                        )
                    else:
                        label = f"{node.name}\nFile does not exist"
                        dot.node(
                            node_id,
                            label,
                            fillcolor="#FFB6C1",  # Light pink
                            color="#FF0000",  # Red border
                            penwidth="3.0",
                            style="filled,dashed",
                        )
                else:
                    dot.node(
                        node_id,
                        node.name,
                        fillcolor="#FFB6C1",  # Light pink
                        color="#333333",
                        penwidth="2.0",
                    )
            elif node.type == NodeType.DATABASE:
                # Special styling for database nodes
                db_type = node.metadata.get("db_type", "unknown")
                label = f"{node.name}\nType: {db_type}"  # Using node.name, not node_id
                dot.node(
                    node_id,
                    label,
                    fillcolor="#B19CD9",  # Light purple for databases
                    color="#333333",
                    penwidth="2.0",
                    shape="cylinder",  # Database shape
                )
            elif node.type == NodeType.INTERNAL_MODULE:
                # Handle imported item nodes with specific style
                if "imported_name" in node.metadata:
                    dot.node(
                        node_id,
                        node.name,
                        fillcolor="#ADD8E6",  # Light blue for internal modules
                        color="#333333",
                        penwidth="2.0",
                        shape="oval",  # Use oval shape for imported items
                    )
                else:
                    dot.node(
                        node_id,
                        node.name,
                        fillcolor="#ADD8E6",  # Light blue for internal modules
                        color="#333333",
                        penwidth="2.0",
                    )
            elif node.type == NodeType.EXTERNAL_MODULE:
                # Handle imported item nodes with specific style
                if "imported_name" in node.metadata:
                    dot.node(
                        node_id,
                        node.name,
                        fillcolor="#FFA07A",  # Light salmon for external modules
                        color="#333333",
                        penwidth="2.0",
                        shape="oval",  # Use oval shape for imported items
                    )
                else:
                    dot.node(
                        node_id,
                        node.name,
                        fillcolor="#FFA07A",  # Light salmon for external modules
                        color="#333333",
                        penwidth="2.0",
                    )
            elif node.type == NodeType.QUARTO_DOCUMENT:
                # Special styling for Quarto documents
                dot.node(
                    node_id,
                    node.name,
                    fillcolor="#00CED1",  # Dark turquoise for Quarto docs
                    color="#333333",
                    penwidth="2.0",
                )

        # Add edges
        dot.attr("edge", color="#333333")
        for edge in model.edges:
            dot.edge(edge.source, edge.target)

        # Render the graph
        output_dir = os.path.dirname(output_path) or "."
        os.makedirs(output_dir, exist_ok=True)

        dot.render(output_path, view=False, format="pdf", cleanup=True)
        print(f"Graphviz visualisation saved as {output_path}.pdf")


class MermaidReporter(Reporter):
    """Generate a Mermaid visualisation of the project model."""

    def generate_report(
        self, model: ProjectModel, output_path: Optional[str] = None
    ) -> None:
        """Generate a Mermaid diagram from the project model."""
        if not output_path:
            output_path = "project_diagram.md"

        # Generate Mermaid markup
        mermaid = [
            "graph TD",
            "    %% Style definitions",
            "    classDef scriptNode fill:#90EE90,stroke:#333,stroke-width:2px;",
            "    classDef fileNode fill:#FFB6C1,stroke:#333,stroke-width:2px;",
            "    classDef quartoNode fill:#00CED1,stroke:#333,stroke-width:2px;",
            "    classDef missingFile fill:#FFB6C1,stroke:#FF0000,stroke-width:3px,stroke-dasharray: 5 5;",
            "    classDef internalModule fill:#ADD8E6,stroke:#333,stroke-width:2px;",
            "    classDef externalModule fill:#FFA07A,stroke:#333,stroke-width:2px;",
            "    classDef importedItem fill:#ADD8E6,stroke:#333,stroke-width:2px,shape:circle;",
            "    classDef externalImportedItem fill:#FFA07A,stroke:#333,stroke-width:2px,shape:circle;",
            "    classDef databaseNode fill:#B19CD9,stroke:#333,stroke-width:2px,shape:cylinder;",
            "",
            "    %% Nodes",
        ]

        # Add nodes
        for node_id, node in model.nodes.items():
            if node.type == NodeType.SCRIPT:
                mermaid.append(f'    {node_id}["{node.name}"]:::scriptNode')
            elif node.type == NodeType.DATA_FILE:
                # Handle file status for data files
                if "status" in node.metadata:
                    status = node.metadata["status"]
                    if status.exists:
                        mod_time = status.last_modified.strftime("%Y-%m-%d %H:%M:%S")
                        label = f"{node.name}<br/><small>Modified: {mod_time}</small>"
                        mermaid.append(f'    {node_id}["{label}"]:::fileNode')
                    else:
                        label = f"{node.name}<br/><small>File does not exist</small>"
                        mermaid.append(f'    {node_id}["{label}"]:::missingFile')
                else:
                    mermaid.append(f'    {node_id}["{node.name}"]:::fileNode')
            elif node.type == NodeType.DATABASE:
                # Database nodes with specific styling
                db_type = node.metadata.get("db_type", "unknown")
                label = f"{node.name}<br/><small>Type: {db_type}</small>"
                mermaid.append(f'    {node_id}["{label}"]:::databaseNode')
            elif node.type == NodeType.INTERNAL_MODULE:
                # Handle imported item nodes with specific style
                if "imported_name" in node.metadata:
                    mermaid.append(f'    {node_id}(("{node.name}")):::importedItem')
                else:
                    mermaid.append(f'    {node_id}["{node.name}"]:::internalModule')
            elif node.type == NodeType.EXTERNAL_MODULE:
                # Handle imported item nodes with specific style
                if "imported_name" in node.metadata:
                    mermaid.append(
                        f'    {node_id}(("{node.name}")):::externalImportedItem'
                    )
                else:
                    mermaid.append(f'    {node_id}["{node.name}"]:::externalModule')
            elif node.type == NodeType.QUARTO_DOCUMENT:
                mermaid.append(f'    {node_id}["{node.name}"]:::quartoNode')

        mermaid.append("")
        mermaid.append("    %% Relationships")

        # Add edges
        for edge in model.edges:
            mermaid.append(f"    {edge.source} --> {edge.target}")

        # Create markdown file with mermaid diagram
        output_dir = os.path.dirname(output_path) or "."
        os.makedirs(output_dir, exist_ok=True)

        with open(output_path, "w") as f:
            f.write("# Project Dependency Diagram\n\n")
            f.write("```mermaid\n")
            f.write("\n".join(mermaid))
            f.write("\n```\n")

        print(f"Mermaid diagram saved as {output_path}")


class JsonReporter(Reporter):
    """Generate a JSON representation of the project model."""

    def generate_report(
        self, model: ProjectModel, output_path: Optional[str] = None
    ) -> None:
        """Generate a JSON file from the project model or print to console if no path is given."""
        # Create a serializable representation of the model
        serializable = {"nodes": [], "edges": [], "file_operations": [], "imports": []}

        # Add nodes
        for node_id, node in model.nodes.items():
            node_data = {
                "id": node_id,
                "name": node.name,
                "type": node.type,
                "metadata": {},
            }

            # Handle file status for data files
            if node.type == NodeType.DATA_FILE and "status" in node.metadata:
                status = node.metadata["status"]
                node_data["metadata"]["exists"] = status.exists
                if status.last_modified:
                    node_data["metadata"]["last_modified"] = (
                        status.last_modified.isoformat()
                    )

            serializable["nodes"].append(node_data)

        # Add edges
        for edge in model.edges:
            serializable["edges"].append(
                {"source": edge.source, "target": edge.target, "type": edge.type}
            )

        # Add file operations
        for filename, operations in model.file_operations.items():
            for op in operations:
                serializable["file_operations"].append(
                    {
                        "filename": op.filename,
                        "is_read": op.is_read,
                        "is_write": op.is_write,
                        "source_file": op.source_file,
                    }
                )

        # Add imports
        for source_file, imports in model.imports.items():
            for imp in imports:
                serializable["imports"].append(
                    {
                        "module_name": imp.module_name,
                        "source_file": imp.source_file,
                        "is_from_import": imp.is_from_import,
                        "imported_names": imp.imported_names,
                        "is_internal": imp.is_internal,
                    }
                )

        # If no output path specified, print to console with rich
        if output_path is None:
            console = Console()
            console.print("\n[bold cyan]JSON Representation[/bold cyan]")
            console.print("=" * 80)
            console.print_json(data=serializable, indent=2)
        else:
            # Write to file
            output_dir = os.path.dirname(output_path) or "."
            os.makedirs(output_dir, exist_ok=True)

            with open(output_path, "w") as f:
                json.dump(serializable, f, indent=2)

            print(f"JSON report saved as {output_path}")


def get_reporter(format_type: str) -> Reporter:
    """
    Factory function to get the appropriate reporter.

    Args:
        format_type: The type of reporter to use ('console', 'graphviz', 'mermaid', or 'json')

    Returns:
        A Reporter instance

    Raises:
        ValueError: If the format type is not supported
    """
    reporters = {
        "console": ConsoleReporter(),
        "graphviz": GraphvizReporter(),
        "mermaid": MermaidReporter(),
        "json": JsonReporter(),
    }

    if format_type.lower() not in reporters:
        raise ValueError(
            f"Unsupported format: {format_type}. "
            f"Supported formats: {', '.join(reporters.keys())}"
        )

    return reporters[format_type.lower()]
