"""Tests for CLI commands."""

from __future__ import annotations

from click.testing import CliRunner

from promptgauntlet.cli import main


class TestCLI:
    """Tests for CLI subcommands."""

    def test_version(self) -> None:
        runner = CliRunner()
        result = runner.invoke(main, ["--version"])
        assert result.exit_code == 0
        assert "0.1.0" in result.output

    def test_list_all(self) -> None:
        runner = CliRunner()
        result = runner.invoke(main, ["list"])
        assert result.exit_code == 0
        assert "CLASSIFICATION" in result.output
        assert "CONSTRAINT" in result.output
        assert "TOOL_USE" in result.output
        assert "CONVERGENCE" in result.output

    def test_list_filtered(self) -> None:
        runner = CliRunner()
        result = runner.invoke(main, ["list", "--family", "constraint"])
        assert result.exit_code == 0
        assert "json_schema" in result.output
        assert "CLASSIFICATION" not in result.output

    def test_help(self) -> None:
        runner = CliRunner()
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "list" in result.output
        assert "run" in result.output
        assert "human" in result.output
        assert "grade" in result.output
        assert "report" in result.output

    def test_run_mock(self) -> None:
        """Run with mock model succeeds."""
        runner = CliRunner()
        result = runner.invoke(
            main,
            ["run", "--model", "mock", "--scenarios", "all", "--seeds", "1"],
        )
        assert result.exit_code == 0
        assert "Run complete" in result.output
