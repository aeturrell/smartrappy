"""Code analyser for smartrappy."""

import ast
import os
from typing import List, Optional, Set, Tuple

from smartrappy.models import DatabaseInfo, FileInfo, ModuleImport, ProjectModel


def get_mode_properties(mode: str) -> tuple[bool, bool]:
    """
    Determine read/write properties from a file mode string.

    Args:
        mode: File mode string (e.g., 'r', 'w', 'a', 'x', 'r+', etc.)

    Returns:
        Tuple of (is_read, is_write)
    """
    # Default mode 'r' if not specified
    mode = mode or "r"

    # Plus sign adds read & write capabilities
    if "+" in mode:
        return True, True

    # Basic mode mapping
    mode_map = {
        "r": (True, False),  # read only
        "w": (False, True),  # write only (truncate)
        "a": (False, True),  # write only (append)
        "x": (False, True),  # write only (exclusive creation)
    }

    # Get base mode (first character)
    base_mode = mode[0]
    return mode_map.get(base_mode, (False, False))


def get_open_file_info(node: ast.Call, source_file: str) -> Optional[FileInfo]:
    """Extract file information from an open() function call."""
    if not (isinstance(node.func, ast.Name) and node.func.id == "open"):
        return None

    # Get filename from first argument
    if not (len(node.args) > 0 and isinstance(node.args[0], ast.Str)):
        return None

    filename = node.args[0].s

    # Default mode is 'r'
    mode = "r"

    # Check positional mode argument
    if len(node.args) > 1 and isinstance(node.args[1], ast.Str):
        mode = node.args[1].s

    # Check for mode in keyword arguments
    for keyword in node.keywords:
        if keyword.arg == "mode" and isinstance(keyword.value, ast.Str):
            mode = keyword.value.s

    is_read, is_write = get_mode_properties(mode)

    return FileInfo(
        filename=filename, is_read=is_read, is_write=is_write, source_file=source_file
    )


def get_pandas_file_info(node: ast.Call, source_file: str) -> Optional[FileInfo]:
    """Extract file information from pandas operations (both pd.read_* and DataFrame writes)."""
    # Case 1: pd.read_* or pd.to_* function calls
    if isinstance(node.func, ast.Attribute):
        if hasattr(node.func.value, "id"):
            # Direct pandas import calls (pd.read_csv, etc.)
            if node.func.value.id == "pd":
                # Skip SQL-related functions that don't read files
                if node.func.attr in ["read_sql", "read_sql_query", "read_sql_table"]:
                    return None

                if not (len(node.args) > 0 and isinstance(node.args[0], ast.Str)):
                    return None

                filename = node.args[0].s
                method = node.func.attr

                is_read = method.startswith("read_")
                is_write = method.startswith("to_")

                if not (is_read or is_write):
                    return None

                return FileInfo(
                    filename=filename,
                    is_read=is_read,
                    is_write=is_write,
                    source_file=source_file,
                )

        # DataFrame method calls (df.to_csv, etc.)
        method = node.func.attr
        if method.startswith("to_"):
            # Skip to_sql method as it writes to a database, not a file
            if method == "to_sql":
                return None

            if not (len(node.args) > 0 and isinstance(node.args[0], ast.Str)):
                return None

            filename = node.args[0].s
            return FileInfo(
                filename=filename, is_read=False, is_write=True, source_file=source_file
            )

    return None


def get_matplotlib_file_info(node: ast.Call, source_file: str) -> Optional[FileInfo]:
    """Extract file information from matplotlib save operations."""
    if not isinstance(node.func, ast.Attribute):
        return None

    # Check if it's a savefig call
    if node.func.attr != "savefig":
        return None

    # Handle both plt.savefig() and Figure.savefig()
    if hasattr(node.func.value, "id"):
        if node.func.value.id not in ["plt", "fig", "figure"]:
            return None

    # Get filename from first argument or fname keyword
    filename = None

    # Check positional argument
    if len(node.args) > 0 and isinstance(node.args[0], ast.Str):
        filename = node.args[0].s

    # Check for fname keyword argument
    for keyword in node.keywords:
        if keyword.arg == "fname" and isinstance(keyword.value, ast.Str):
            filename = keyword.value.s

    if not filename:
        return None

    return FileInfo(
        filename=filename, is_read=False, is_write=True, source_file=source_file
    )


class FileOperationFinder(ast.NodeVisitor):
    """AST visitor that finds file operations in Python code."""

    def __init__(self, source_file: str):
        self.source_file = source_file
        self.file_operations: List[FileInfo] = []

    def visit_Call(self, node: ast.Call):
        # Check for open() calls
        if file_info := get_open_file_info(node, self.source_file):
            self.file_operations.append(file_info)

        # Check for pandas operations
        if file_info := get_pandas_file_info(node, self.source_file):
            self.file_operations.append(file_info)

        # Check for matplotlib operations
        if file_info := get_matplotlib_file_info(node, self.source_file):
            self.file_operations.append(file_info)

        self.generic_visit(node)


class ModuleImportFinder(ast.NodeVisitor):
    """AST visitor that finds module imports in Python code."""

    def __init__(self, source_file: str, project_modules: Set[str]):
        self.source_file = source_file
        self.project_modules = project_modules
        self.imports: List[ModuleImport] = []

    def visit_Import(self, node: ast.Import):
        for name in node.names:
            base_module = name.name.split(".")[0]
            self.imports.append(
                ModuleImport(
                    module_name=name.name,
                    source_file=self.source_file,
                    is_from_import=False,
                    imported_names=[name.asname or name.name],
                    is_internal=base_module in self.project_modules,
                )
            )

    def visit_ImportFrom(self, node: ast.ImportFrom):
        if node.module:  # Ignore relative imports for simplicity
            base_module = node.module.split(".")[0]
            imported_names = [name.name for name in node.names]
            self.imports.append(
                ModuleImport(
                    module_name=node.module,
                    source_file=self.source_file,
                    is_from_import=True,
                    imported_names=imported_names,
                    is_internal=base_module in self.project_modules,
                )
            )


class DatabaseOperationFinder(ast.NodeVisitor):
    """AST visitor that finds database operations in Python code."""

    def __init__(self, source_file: str):
        self.source_file = source_file
        self.database_operations: List[DatabaseInfo] = []
        # Track connection variables and their database info
        self.connection_variables = {}  # Map from variable name to DatabaseInfo
        self.current_assignment_target = None
        self.sqlalchemy_engines = {}  # Track SQLAlchemy engines

    def visit_Assign(self, node: ast.Assign):
        # Track the current assignment target
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            self.current_assignment_target = node.targets[0].id
            self.generic_visit(node)
            self.current_assignment_target = None
        else:
            self.generic_visit(node)

    def visit_Call(self, node: ast.Call):
        # Check for SQLAlchemy engine creation
        if (
            isinstance(node.func, ast.Attribute)
            and hasattr(node.func.value, "id")
            and node.func.value.id == "create_engine"
            and self.current_assignment_target
        ):
            # This is a SQLAlchemy engine being created
            db_info = get_sqlalchemy_info(node, self.source_file)
            if db_info:
                # Register this engine variable
                self.sqlalchemy_engines[self.current_assignment_target] = db_info
                # Store with engine variable name
                db_info_with_var = db_info._replace(
                    conn_var_name=self.current_assignment_target
                )
                self.database_operations.append(db_info_with_var)

        # Check for SQLAlchemy function calls themselves
        elif isinstance(node.func, ast.Name) and node.func.id == "create_engine":
            # Direct create_engine call
            db_info = get_sqlalchemy_info(node, self.source_file)
            if db_info and self.current_assignment_target:
                # Register this engine variable
                self.sqlalchemy_engines[self.current_assignment_target] = db_info
                # Store with engine variable name
                db_info_with_var = db_info._replace(
                    conn_var_name=self.current_assignment_target
                )
                self.database_operations.append(db_info_with_var)

        # Check for database connection creation
        elif isinstance(node.func, ast.Attribute) and node.func.attr == "connect":
            # This is a database connection being created
            db_info = get_direct_db_driver_info(node, self.source_file)
            if db_info and self.current_assignment_target:
                # Register this connection variable
                self.connection_variables[self.current_assignment_target] = db_info
                # Store with connection variable name
                db_info_with_var = db_info._replace(
                    conn_var_name=self.current_assignment_target
                )
                self.database_operations.append(db_info_with_var)

        # Check for pandas SQL operations using connection variables or SQLAlchemy engines
        elif isinstance(node.func, ast.Attribute) and node.func.attr in [
            "read_sql",
            "read_sql_query",
            "read_sql_table",
        ]:
            # Find the connection variable or engine
            conn_var = None

            # Check in positional args (usually the second argument)
            if len(node.args) > 1 and isinstance(node.args[1], ast.Name):
                conn_var = node.args[1].id

            # Check in keywords
            for kw in node.keywords:
                if kw.arg == "con" and isinstance(kw.value, ast.Name):
                    conn_var = kw.value.id

            # First check if it's a known SQLAlchemy engine
            if conn_var and conn_var in self.sqlalchemy_engines:
                orig_db_info = self.sqlalchemy_engines[conn_var]
                # Create a new operation with read access only
                read_db_info = DatabaseInfo(
                    db_name=orig_db_info.db_name,
                    connection_string=orig_db_info.connection_string,
                    db_type=orig_db_info.db_type,
                    is_read=True,
                    is_write=False,
                    source_file=self.source_file,
                    uses_conn_var=conn_var,
                )
                self.database_operations.append(read_db_info)
            # Then check if it's a known database connection
            elif conn_var and conn_var in self.connection_variables:
                # We found a pandas SQL operation using a known connection
                orig_db_info = self.connection_variables[conn_var]
                # Create a new operation with read access only
                read_db_info = DatabaseInfo(
                    db_name=orig_db_info.db_name,
                    connection_string=orig_db_info.connection_string,
                    db_type=orig_db_info.db_type,
                    is_read=True,
                    is_write=False,
                    source_file=self.source_file,
                    uses_conn_var=conn_var,
                )
                self.database_operations.append(read_db_info)
            else:
                # Connection variable not found or operation doesn't use a variable
                # Process as generic pandas SQL operation
                db_info = get_pandas_sql_info(node, self.source_file)
                if db_info:
                    self.database_operations.append(db_info)

        # Check for DataFrame.to_sql method (write operation)
        elif isinstance(node.func, ast.Attribute) and node.func.attr == "to_sql":
            # Find the connection variable or engine
            conn_var = None

            # Check in positional args (usually the second argument)
            if len(node.args) > 1 and isinstance(node.args[1], ast.Name):
                conn_var = node.args[1].id

            # Check in keywords
            for kw in node.keywords:
                if kw.arg == "con" and isinstance(kw.value, ast.Name):
                    conn_var = kw.value.id

            # First check if it's a known SQLAlchemy engine
            if conn_var and conn_var in self.sqlalchemy_engines:
                orig_db_info = self.sqlalchemy_engines[conn_var]

                # # Get table name if available
                # table_name = "unknown_table"
                # if len(node.args) > 0 and isinstance(node.args[0], ast.Str):
                #     table_name = node.args[0].s

                # Create a new operation with write access only
                write_db_info = DatabaseInfo(
                    db_name=orig_db_info.db_name,
                    connection_string=orig_db_info.connection_string,
                    db_type=orig_db_info.db_type,
                    is_read=False,
                    is_write=True,
                    source_file=self.source_file,
                    uses_conn_var=conn_var,
                )
                self.database_operations.append(write_db_info)
            # Then check if it's a known database connection
            elif conn_var and conn_var in self.connection_variables:
                # We found a to_sql operation using a known connection
                orig_db_info = self.connection_variables[conn_var]

                # # Get table name if available
                # table_name = "unknown_table"
                # if len(node.args) > 0 and isinstance(node.args[0], ast.Str):
                #     table_name = node.args[0].s

                # Create a new operation with write access only
                write_db_info = DatabaseInfo(
                    db_name=orig_db_info.db_name,
                    connection_string=orig_db_info.connection_string,
                    db_type=orig_db_info.db_type,
                    is_read=False,
                    is_write=True,
                    source_file=self.source_file,
                    uses_conn_var=conn_var,
                )
                self.database_operations.append(write_db_info)
            else:
                # Connection variable not found or operation doesn't use a variable
                # Process as generic pandas to_sql operation
                db_info = get_pandas_sql_info(node, self.source_file)
                if db_info:
                    self.database_operations.append(db_info)
        else:
            # Check for other database operations
            db_info = None

            # Check for SQLAlchemy operations
            if db_info is None:
                db_info = get_sqlalchemy_info(node, self.source_file)

            # If not SQLAlchemy, try regular pandas operations
            if db_info is None and not (
                isinstance(node.func, ast.Attribute)
                and node.func.attr
                in ["read_sql", "read_sql_query", "read_sql_table", "to_sql"]
            ):
                db_info = get_pandas_sql_info(node, self.source_file)

            if db_info:
                self.database_operations.append(db_info)

        self.generic_visit(node)


def get_pandas_sql_info(node: ast.Call, source_file: str) -> Optional[DatabaseInfo]:
    """Extract database information from pandas SQL operations."""
    if not isinstance(node.func, ast.Attribute):
        return None

    # Check for pandas read_sql operations
    if node.func.attr in ["read_sql", "read_sql_query", "read_sql_table"]:
        # Extract connection from arguments (often 2nd argument)
        conn_string = None
        db_name = "pandas_sql_db"
        db_type = "unknown"
        conn_var_name = None

        # Check for connection in args or kwargs
        conn_arg_idx = 1  # Default position for connection in read_sql functions

        # Check positional arguments for connection
        if len(node.args) > conn_arg_idx:
            if isinstance(node.args[conn_arg_idx], ast.Name):
                # Connection is provided as a variable
                conn_var_name = node.args[conn_arg_idx].id

        # Check for connection in kwargs
        for keyword in node.keywords:
            if keyword.arg == "con":
                # Connection can be a string or a connection object
                if isinstance(keyword.value, ast.Str):
                    conn_string = keyword.value.s

                    # Attempt to determine database type from connection string
                    if "postgresql" in conn_string.lower():
                        db_type = "postgresql"
                    elif "mysql" in conn_string.lower():
                        db_type = "mysql"
                    elif "sqlite" in conn_string.lower():
                        db_type = "sqlite"
                    elif any(
                        x in conn_string.lower()
                        for x in ["mssql", "sql server", "sqlserver", "odbc"]
                    ):
                        db_type = "mssql"

                    # Extract DB name if possible
                    import re

                    # Common patterns for database names in connection strings
                    patterns = [
                        r"/([^/]+)$",  # Standard URI format
                        r"database=([^;]+)",  # MSSQL/ODBC style
                        r"initial catalog=([^;]+)",  # MSSQL style
                    ]

                    for pattern in patterns:
                        match = re.search(pattern, conn_string, re.IGNORECASE)
                        if match:
                            db_name = match.group(1)
                            break
                elif isinstance(keyword.value, ast.Name):
                    # Store the connection variable name for later connection tracking
                    conn_var_name = keyword.value.id

        # Store the connection variable name in the metadata
        # This will be used later to correlate with actual database connections
        metadata = {}
        if conn_var_name:
            metadata["conn_var_name"] = conn_var_name

        return DatabaseInfo(
            db_name=db_name,
            connection_string=conn_string,
            db_type=db_type,
            is_read=True,
            is_write=False,
            source_file=source_file,
        )

    # Check for DataFrame.to_sql method
    elif node.func.attr == "to_sql":
        # This is a write operation
        table_name = None
        db_name = "pandas_sql_db"
        db_type = "unknown"

        # Check for table name in first arg
        if len(node.args) > 0 and isinstance(node.args[0], ast.Str):
            table_name = node.args[0].s
            db_name = f"pandas_sql_db:{table_name}"

        # Check for connection in args or kwargs
        conn_string = None
        for keyword in node.keywords:
            if keyword.arg == "con" and isinstance(keyword.value, ast.Str):
                conn_string = keyword.value.s

                # Attempt to determine database type from connection string
                if "postgresql" in conn_string.lower():
                    db_type = "postgresql"
                elif "mysql" in conn_string.lower():
                    db_type = "mysql"
                elif "sqlite" in conn_string.lower():
                    db_type = "sqlite"

                # Extract DB name if possible from connection string
                import re

                patterns = [
                    r"/([^/]+)$",  # Standard URI format
                    r"database=([^;]+)",  # MSSQL/ODBC style
                    r"initial catalog=([^;]+)",  # MSSQL style
                ]
                for pattern in patterns:
                    match = re.search(pattern, conn_string, re.IGNORECASE)
                    if match:
                        db_name = match.group(1)
                        break

        return DatabaseInfo(
            db_name=db_name,
            connection_string=conn_string,
            db_type=db_type,
            is_read=False,
            is_write=True,
            source_file=source_file,
        )

    return None


def get_sqlalchemy_info(node: ast.Call, source_file: str) -> Optional[DatabaseInfo]:
    """Extract database information from SQLAlchemy operations."""
    # Check for create_engine calls
    if not (
        (isinstance(node.func, ast.Attribute) and node.func.attr == "create_engine")
        or (isinstance(node.func, ast.Name) and node.func.id == "create_engine")
    ):
        return None

    # Extract connection string from the first argument
    if len(node.args) > 0 and isinstance(node.args[0], ast.Str):
        conn_string = node.args[0].s

        # Try to extract database type and name from connection string
        db_type = "unknown"
        db_name = "unknown_db"

        # Parse common SQLAlchemy connection strings
        if conn_string.startswith("postgresql"):
            db_type = "postgresql"
            # Extract database name from connection string
            import re

            match = re.search(r"/([^/]+)$", conn_string)
            if match:
                db_name = match.group(1)
        elif conn_string.startswith("mysql"):
            db_type = "mysql"
            # Extract database name
            import re

            match = re.search(r"/([^/]+)$", conn_string)
            if match:
                db_name = match.group(1)
        elif conn_string.startswith("sqlite"):
            db_type = "sqlite"
            # For SQLite, the database name is the file path
            import re

            match = re.search(r"sqlite:///(.+)$", conn_string)
            if match:
                db_name = match.group(1)
        elif any(x in conn_string.lower() for x in ["mssql", "pyodbc", "pymssql"]):
            # Handle MS SQL Server connection strings
            db_type = "mssql"
            import re

            # Look for database name in different format variations
            patterns = [
                r"database=([^;]+)",
                r"initial catalog=([^;]+)",
                r"/([^/]+)$",  # For URLs like mssql+pyodbc://server/database
            ]

            for pattern in patterns:
                match = re.search(pattern, conn_string, re.IGNORECASE)
                if match:
                    db_name = match.group(1)
                    break

        # We can't determine read/write at the create_engine level
        # Default to both since we don't know the actual operations
        return DatabaseInfo(
            db_name=db_name,
            connection_string=conn_string,
            db_type=db_type,
            is_read=True,  # Assuming engine could be used for either
            is_write=True,  # Assuming engine could be used for either
            source_file=source_file,
        )

    return None


def get_direct_db_driver_info(
    node: ast.Call, source_file: str
) -> Optional[DatabaseInfo]:
    """Extract database information from direct database driver calls."""
    if not (isinstance(node.func, ast.Attribute) and node.func.attr == "connect"):
        return None

    # Extract database type and name
    db_type = "unknown"
    db_name = "unknown_db"

    # Try to determine database type from module
    if hasattr(node.func.value, "id"):
        if node.func.value.id in ["psycopg2", "psycopg"]:
            db_type = "postgresql"
            db_name = "postgresql_db"
        elif node.func.value.id == "sqlite3":
            db_type = "sqlite"
            # SQLite databases are files, check for database path
            if len(node.args) > 0 and isinstance(node.args[0], ast.Str):
                db_name = node.args[0].s
        elif node.func.value.id in ["pyodbc", "pymssql"]:
            db_type = "mssql"
            db_name = "mssql_db"
        elif "mysql" in getattr(node.func.value, "id", ""):
            db_type = "mysql"
            db_name = "mysql_db"

        # Check connection string or parameters for database name
        if len(node.args) > 0 and isinstance(node.args[0], ast.Str):
            conn_string = node.args[0].s
            # Extract database name from connection string
            import re

            patterns = [
                r"database=([^;]+)",
                r"initial catalog=([^;]+)",
                r"dbname=([^;]+)",
            ]
            for pattern in patterns:
                match = re.search(pattern, conn_string, re.IGNORECASE)
                if match:
                    db_name = match.group(1)
                    break

        # Check for database parameter in keywords
        for keyword in node.keywords:
            if keyword.arg in ["database", "db", "dbname"] and isinstance(
                keyword.value, ast.Str
            ):
                db_name = keyword.value.s

    return DatabaseInfo(
        db_name=db_name,
        connection_string=None,
        db_type=db_type,
        is_read=True,  # Connection creation is both
        is_write=True,  # Connection creation is both
        source_file=source_file,
    )


def get_project_modules(folder_path: str) -> Set[str]:
    """Find all potential internal module names in the project."""
    modules = set()
    for root, dirs, files in os.walk(folder_path):
        # Skip hidden directories (starting with .)
        dirs[:] = [d for d in dirs if not d.startswith(".")]

        for file in files:
            # Skip hidden files (starting with .)
            if file.startswith(".") or not file.endswith(".py"):
                continue

            # Get module name from file path
            rel_path = os.path.relpath(os.path.join(root, file), folder_path)
            module_name = os.path.splitext(rel_path)[0].replace(os.path.sep, ".")
            modules.add(module_name.split(".")[0])  # Add base module name
    return modules


def analyse_python_file(
    file_path: str, project_modules: Set[str]
) -> Tuple[List[FileInfo], List[ModuleImport], List[DatabaseInfo]]:
    """Analyse a single Python file for file operations, imports, and database operations."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            tree = ast.parse(f.read())

        # Find file operations
        file_finder = FileOperationFinder(file_path)
        file_finder.visit(tree)

        # Find imports
        import_finder = ModuleImportFinder(file_path, project_modules)
        import_finder.visit(tree)

        # Find database operations
        db_finder = DatabaseOperationFinder(file_path)
        db_finder.visit(tree)

        return (
            file_finder.file_operations,
            import_finder.imports,
            db_finder.database_operations,
        )

    except (SyntaxError, UnicodeDecodeError, IOError) as e:
        print(f"Error processing {file_path}: {str(e)}")
        return [], [], []


def analyse_project(folder_path: str) -> ProjectModel:
    """
    Analyse a project folder and build a comprehensive project model.

    Args:
        folder_path: Path to the folder to analyse

    Returns:
        A ProjectModel containing the complete analysis results
    """
    model = ProjectModel(folder_path)
    project_modules = get_project_modules(folder_path)

    # Analyse all Python files in the project
    for root, dirs, files in os.walk(folder_path):
        # Skip hidden directories (starting with .)
        dirs[:] = [d for d in dirs if not d.startswith(".")]

        for file in files:
            # Skip hidden files (starting with .)
            if file.startswith(".") or not file.endswith(".py"):
                continue

            file_path = os.path.join(root, file)
            operations, imports, db_operations = analyse_python_file(
                file_path, project_modules
            )

            # Add file operations to the model
            for op in operations:
                model.add_file_operation(op)

            # Add imports to the model
            for imp in imports:
                model.add_import(imp)

            # Add database operations to the model
            for db_op in db_operations:
                model.add_database_operation(db_op)

    # Build the graph representation
    model.build_graph()

    return model
