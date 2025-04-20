"""QMD Parser for smartrappy."""

import ast
import os
import re
from typing import List, Optional, Set, Tuple

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


def analyse_qmd_file(
    file_path: str, project_modules: Set[str], 
    FileOperationFinder, ModuleImportFinder, DatabaseOperationFinder
) -> Tuple[List[FileInfo], List[ModuleImport], List[DatabaseInfo]]:
    """
    Analyse a Quarto markdown file for Python code chunks.
    
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
        
        # Initialize result lists
        all_file_ops = []
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
                print(f"Syntax error in Python chunk {i+1} of {file_path}: {str(e)}")
        
        return all_file_ops, all_imports, all_db_ops
    
    except (UnicodeDecodeError, IOError) as e:
        print(f"Error processing QMD file {file_path}: {str(e)}")
        return [], [], []
