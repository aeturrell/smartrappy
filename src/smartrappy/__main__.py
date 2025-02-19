"""smartrappy"""

import os
import sys
from datetime import datetime

import click

from smartrappy import __version__, analyze_and_visualize


def validate_repo_path(ctx, param, value):
    """Validate that the input path exists and is a directory"""
    if not os.path.exists(value):
        raise click.BadParameter(f"Path does not exist: {value}")
    if not os.path.isdir(value):
        raise click.BadParameter(f"Path is not a directory: {value}")
    return value


def validate_output_path(ctx, param, value):
    """Validate that the output path is writable"""
    if value is None:
        return None

    try:
        directory = os.path.dirname(value) or "."
        if not os.path.exists(directory):
            os.makedirs(directory)
        # Check if we can write to this location
        test_file = f"{value}_test"
        with open(test_file, "w") as f:
            f.write("")
        os.remove(test_file)
        return value
    except (OSError, IOError) as e:
        raise click.BadParameter(f"Cannot write to output location: {value}\n{str(e)}")


@click.command(context_settings=dict(help_option_names=["-h", "--help"]))
@click.argument(
    "repo_path",
    callback=validate_repo_path,
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
)
@click.option(
    "-o",
    "--output",
    callback=validate_output_path,
    help="Output path for the analysis files (without extension)",
    type=click.Path(dir_okay=False),
)
@click.version_option(version=__version__, prog_name="smartrappy")
def main(repo_path, output):
    """Smart reproducible analytical pipeline execution analyzer.

    Analyzes Python projects to create a visual representation of file operations
    and module dependencies.

    Examples:

    \b
    # Analyze current directory with default output
    smartrappy .

    \b
    # Analyze specific project with custom output location
    smartrappy /path/to/project -o /path/to/output/analysis
    """
    # Generate default output path if none provided
    if output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output = os.path.join(repo_path, f"smartrappy_analysis_{timestamp}")

    try:
        analyze_and_visualize(repo_path, output)
    except Exception as e:
        click.secho(f"Error during analysis: {str(e)}", fg="red", err=True)
        sys.exit(1)


if __name__ == "__main__":
    main(prog_name="smartrappy")  # pragma: no cover
