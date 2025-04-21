"""Command-line interface for smartrappy."""

import os
import sys
from datetime import datetime

import click

from smartrappy import __version__
from smartrappy.analyser import analyse_project
from smartrappy.reporters import get_reporter


def validate_repo_path(ctx, param, value):
    """Validate that the input path exists and is a directory."""
    if not os.path.exists(value):
        raise click.BadParameter(f"Path does not exist: {value}")
    if not os.path.isdir(value):
        raise click.BadParameter(f"Path is not a directory: {value}")
    return value


def validate_output_path(ctx, param, value):
    """Validate that the output path is writable."""
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
@click.option(
    "-f",
    "--format",
    "format_type",
    type=click.Choice(["console", "graphviz", "mermaid", "json"], case_sensitive=False),
    default="console",
    help="Output format for the analysis (default: console)",
)
@click.option(
    "--all-formats",
    is_flag=True,
    help="Generate all output formats",
)
@click.option(
    "--internal",
    is_flag=True,
    help="Only include internal modules in the visualisation (exclude external packages)",
)
@click.version_option(version=__version__, prog_name="smartrappy")
def main(repo_path, output, format_type, all_formats, internal):
    """Smart reproducible analytical pipeline execution analyser.

    Analyses Python projects to create a visual representation of file operations
    and module dependencies.

    Examples:

    \b
    # Analyse current directory with default console output
    smartrappy .

    \b
    # Analyse specific project with graphviz output
    smartrappy /path/to/project --formnat graphviz --output /path/to/output/analysis

    \b
    # Generate all output formats
    smartrappy /path/to/project --all-formats --output /path/to/output/analysis

    \b
    # Show only internal module dependencies
    smartrappy /path/to/project --internal
    """
    try:
        # Analyse the project
        click.echo(f"Analysing project at: {repo_path}")
        model = analyse_project(repo_path, internal_only=internal)

        # Generate reports
        formats_to_generate = (
            ["console", "graphviz", "mermaid", "json"] if all_formats else [format_type]
        )

        for fmt in formats_to_generate:
            try:
                reporter = get_reporter(fmt)

                # Handle output paths based on format
                fmt_output = None

                # Only use output path for formats that need files
                if fmt in ["graphviz", "mermaid"]:
                    # Generate default output path if none provided
                    if output is None:
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        base_output = f"smartrappy_analysis_{timestamp}"
                    else:
                        base_output = output

                    # Append format type to output path when generating multiple formats
                    fmt_output = (
                        f"{base_output}_{fmt}"
                        if len(formats_to_generate) > 1
                        else base_output
                    )

                # For JSON, only use output path if explicitly provided by the user
                elif fmt == "json" and output is not None:
                    fmt_output = (
                        f"{output}_{fmt}" if len(formats_to_generate) > 1 else output
                    )

                reporter.generate_report(model, fmt_output)
            except Exception as e:
                click.secho(
                    f"Error generating {fmt} report: {str(e)}", fg="yellow", err=True
                )

    except Exception as e:
        click.secho(f"Error during analysis: {str(e)}", fg="red", err=True)
        sys.exit(1)


if __name__ == "__main__":
    main(prog_name="smartrappy")  # pragma: no cover
