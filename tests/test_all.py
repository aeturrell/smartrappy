"""Tests for the refactored smartrappy architecture."""

import tempfile
from pathlib import Path

from smartrappy import analyze_project
from smartrappy.reporters import (
    ConsoleReporter,
    GraphvizReporter,
    JsonReporter,
    MermaidReporter,
)


def test_analyze_project():
    """Test that analyze_project works with the test directories."""
    # Analyze the test set
    model = analyze_project("tests/test_set_one")

    # Check that the model contains expected data
    assert len(model.nodes) > 0
    assert len(model.edges) > 0
    assert "data.csv" in model.file_operations

    # Test with a different directory
    model2 = analyze_project("tests/test_set_two")
    assert len(model2.nodes) > 0
    assert "data/input.csv" in model2.file_operations


def test_reporters():
    """Test that all reporters can generate output."""
    # Analyze a test set
    model = analyze_project("tests/test_set_one")

    with tempfile.TemporaryDirectory() as tmpdir:
        # Test console reporter
        console_reporter = ConsoleReporter()
        console_reporter.generate_report(model)  # No output file needed

        # Test graphviz reporter
        graphviz_output = Path(tmpdir) / "graphviz_test"
        graphviz_reporter = GraphvizReporter()
        graphviz_reporter.generate_report(model, str(graphviz_output))
        assert (graphviz_output.with_suffix(".pdf")).exists()

        # Test mermaid reporter
        mermaid_output = Path(tmpdir) / "mermaid_test.md"
        mermaid_reporter = MermaidReporter()
        mermaid_reporter.generate_report(model, str(mermaid_output))
        assert mermaid_output.exists()

        # Test JSON reporter with console output
        json_reporter = JsonReporter()
        json_reporter.generate_report(model)  # Should print to console

        # Test JSON reporter with file output
        json_output = Path(tmpdir) / "json_test.json"
        json_reporter.generate_report(model, str(json_output))
        assert json_output.exists()


if __name__ == "__main__":
    # Simple manual test
    test_analyze_project()
    test_reporters()
    print("All tests passed!")
