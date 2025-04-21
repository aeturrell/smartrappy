import ast
import re
from typing import List, Set, Tuple

from smartrappy.models import DatabaseInfo, FileInfo, ModuleImport


def extract_python_chunks(qmd_content: str) -> List[str]:
    """
    Extract Python code chunks from a Quarto markdown file.

    Args:
        qmd_content: The content of the QMD file as a string

    Returns:
        A list of Python code chunks found in the file
    """
    # Pattern to match Python code chunks in QMD files
    # Matches ```{python} ... ``` blocks, including those with parameters
    pattern = r"```\{python[^}]*\}(.*?)```"

    # Find all matches using re.DOTALL to match across multiple lines
    matches = re.findall(pattern, qmd_content, re.DOTALL)

    # Clean up the chunks (remove leading/trailing whitespace)
    cleaned_chunks = [chunk.strip() for chunk in matches]

    return cleaned_chunks


def extract_markdown_resources(qmd_content: str) -> List[Tuple[str, str]]:
    """
    Extract markdown resource references from a Quarto markdown file.

    Extracts:
    1. Image references: ![alt](/path/to/image.ext)
    2. Include directives: {{< include /path/to/file.ext >}}

    Args:
        qmd_content: The content of the QMD file as a string

    Returns:
        A list of tuples containing (file_path, resource_type)
    """
    resources = []

    # Pattern to match markdown image syntax ![alt](path)
    image_pattern = r"!\[.*?\]\(([^)]+)\)"
    image_matches = re.findall(image_pattern, qmd_content)

    # Pattern to match Quarto include directives {{< include /path/to/file >}} or {{< include /path/to/file param=value >}}
    include_pattern = r"\{\{<\s*include\s+([^\s>]+)(?:\s+[^>]+?)?\s*>\}\}"
    include_matches = re.findall(include_pattern, qmd_content)

    # Process image paths
    for path in image_matches:
        # Remove query parameters if present
        clean_path = path.split("?")[0].strip()
        # Remove any fragment identifiers
        clean_path = clean_path.split("#")[0].strip()
        # Remove any surrounding quotation marks
        if (clean_path.startswith('"') and clean_path.endswith('"')) or (
            clean_path.startswith("'") and clean_path.endswith("'")
        ):
            clean_path = clean_path[1:-1]

        # Ignore external URLs
        if not clean_path.startswith(("http://", "https://", "ftp://")):
            # Remove leading slash if present
            if clean_path.startswith("/"):
                clean_path = clean_path[1:]
            resources.append((clean_path, "image"))

    # Process include directives
    for path in include_matches:
        clean_path = path.strip()

        # The regex might capture additional parameters after the path,
        # so ensure we just get the file path by splitting on whitespace
        # and taking the first part (the file path)
        clean_path = clean_path.split()[0] if " " in clean_path else clean_path

        # Remove any surrounding quotation marks
        if (clean_path.startswith('"') and clean_path.endswith('"')) or (
            clean_path.startswith("'") and clean_path.endswith("'")
        ):
            clean_path = clean_path[1:-1]

        # Remove leading slash if present
        if clean_path.startswith("/"):
            clean_path = clean_path[1:]

        resources.append((clean_path, "include"))

    return resources


def analyse_qmd_file(
    file_path: str,
    project_modules: Set[str],
    FileOperationFinder,
    ModuleImportFinder,
    DatabaseOperationFinder,
) -> Tuple[List[FileInfo], List[ModuleImport], List[DatabaseInfo]]:
    """
    Analyse a Quarto markdown file for Python code chunks and external resources.

    Detects:
    - Python code chunks
    - Markdown image references (![alt](/path/to/image.ext))
    - Quarto include directives ({{< include /path/to/file.ext >}})

    Args:
        file_path: Path to the QMD file
        project_modules: Set of known project module names
        FileOperationFinder: Class to find file operations
        ModuleImportFinder: Class to find module imports
        DatabaseOperationFinder: Class to find database operations

    Returns:
        A tuple of (file_operations, imports, database_operations)
    """
    try:
        # Read the QMD file content
        with open(file_path, "r", encoding="utf-8") as f:
            qmd_content = f.read()

        # Extract Python code chunks
        python_chunks = extract_python_chunks(qmd_content)

        # Extract markdown resources (images and includes)
        resources = extract_markdown_resources(qmd_content)

        # Create FileInfo objects for resource inputs (images, includes, etc.)
        resource_file_ops = []
        for resource_path, resource_type in resources:
            # All external resources are considered read operations in QMD files
            resource_file_ops.append(
                FileInfo(
                    filename=resource_path,
                    is_read=True,
                    is_write=False,
                    source_file=file_path,
                )
            )

        # Initialize result lists
        all_file_ops = resource_file_ops  # Start with external resource operations
        all_imports = []
        all_db_ops = []

        # Process each Python chunk separately
        for i, chunk in enumerate(python_chunks):
            try:
                # Parse the chunk as Python code
                tree = ast.parse(chunk)

                # Find file operations
                file_finder = FileOperationFinder(file_path)
                file_finder.visit(tree)
                all_file_ops.extend(file_finder.file_operations)

                # Find imports
                import_finder = ModuleImportFinder(file_path, project_modules)
                import_finder.visit(tree)
                all_imports.extend(import_finder.imports)

                # Find database operations
                db_finder = DatabaseOperationFinder(file_path)
                db_finder.visit(tree)
                all_db_ops.extend(db_finder.database_operations)

            except SyntaxError as e:
                print(f"Syntax error in Python chunk {i + 1} of {file_path}: {str(e)}")

        return all_file_ops, all_imports, all_db_ops

    except (UnicodeDecodeError, IOError) as e:
        print(f"Error processing QMD file {file_path}: {str(e)}")
        return [], [], []
