"""Integration tests for reporters to improve coverage."""

import json
import os
import tempfile

from smartrappy.analyser import analyse_project
from smartrappy.reporters import (
    ConsoleReporter,
    JsonReporter,
    MermaidReporter,
    get_reporter,
)


def test_console_reporter_with_real_project():
    """Test console reporter with actual project analysis."""
    # Use test_set_one for a simple project
    test_dir = "tests/test_set_one"
    if not os.path.exists(test_dir):
        # Skip if test directory doesn't exist
        return

    model = analyse_project(test_dir, internal_only=False)
    reporter = ConsoleReporter()

    # This should not raise an exception
    reporter.generate_report(model)


def test_mermaid_reporter_with_real_project():
    """Test mermaid reporter with actual project analysis."""
    test_dir = "tests/test_set_one"
    if not os.path.exists(test_dir):
        return

    model = analyse_project(test_dir, internal_only=False)

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = os.path.join(tmpdir, "diagram.md")
        reporter = MermaidReporter()
        reporter.generate_report(model, output_path)

        # Verify the file was created and has content
        assert os.path.exists(output_path)
        with open(output_path, "r") as f:
            content = f.read()
            assert "```mermaid" in content
            assert "graph TD" in content


def test_json_reporter_with_real_project():
    """Test JSON reporter with actual project analysis."""
    test_dir = "tests/test_set_one"
    if not os.path.exists(test_dir):
        return

    model = analyse_project(test_dir, internal_only=False)

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = os.path.join(tmpdir, "output.json")
        reporter = JsonReporter()
        reporter.generate_report(model, output_path)

        # Verify the file was created and is valid JSON
        assert os.path.exists(output_path)
        with open(output_path, "r") as f:
            data = json.load(f)
            assert "nodes" in data
            assert "edges" in data


def test_json_reporter_internal_only():
    """Test JSON reporter with internal_only flag."""
    test_dir = "tests/test_set_one"
    if not os.path.exists(test_dir):
        return

    model = analyse_project(test_dir, internal_only=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = os.path.join(tmpdir, "output.json")
        reporter = JsonReporter()
        reporter.generate_report(model, output_path)

        # Verify the file was created
        assert os.path.exists(output_path)
        with open(output_path, "r") as f:
            data = json.load(f)
            # Should have filtered nodes in internal-only mode
            assert "nodes" in data


def test_get_reporter_factory():
    """Test the reporter factory function."""
    console = get_reporter("console")
    assert isinstance(console, ConsoleReporter)

    mermaid = get_reporter("mermaid")
    assert isinstance(mermaid, MermaidReporter)

    json_rep = get_reporter("json")
    assert isinstance(json_rep, JsonReporter)
