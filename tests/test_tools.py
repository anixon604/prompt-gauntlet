"""Tests for simulated tools."""

from __future__ import annotations

import pytest

from promptgauntlet.tools.calculator import CalculatorTool
from promptgauntlet.tools.filestore import FileStoreTool
from promptgauntlet.tools.search import BM25Index, SearchTool


class TestCalculator:
    """Tests for the safe calculator tool."""

    def test_basic_arithmetic(self) -> None:
        calc = CalculatorTool()
        assert calc.execute({"expression": "2 + 3"}) == "5"
        assert calc.execute({"expression": "10 - 4"}) == "6"
        assert calc.execute({"expression": "3 * 7"}) == "21"

    def test_division(self) -> None:
        calc = CalculatorTool()
        result = calc.execute({"expression": "10 / 3"})
        assert float(result) == pytest.approx(3.333333, rel=1e-3)

    def test_power(self) -> None:
        calc = CalculatorTool()
        assert calc.execute({"expression": "2 ** 10"}) == "1024"

    def test_functions(self) -> None:
        calc = CalculatorTool()
        result = calc.execute({"expression": "sqrt(16)"})
        assert result == "4"

    def test_constants(self) -> None:
        calc = CalculatorTool()
        result = float(calc.execute({"expression": "pi"}))
        assert result == pytest.approx(3.14159, rel=1e-3)

    def test_invalid_expression(self) -> None:
        calc = CalculatorTool()
        with pytest.raises(ValueError):
            calc.execute({"expression": "import os"})

    def test_division_by_zero(self) -> None:
        calc = CalculatorTool()
        with pytest.raises(ValueError, match="Division by zero"):
            calc.execute({"expression": "1 / 0"})

    def test_empty_expression(self) -> None:
        calc = CalculatorTool()
        with pytest.raises(ValueError):
            calc.execute({"expression": ""})


class TestSearch:
    """Tests for the BM25 search tool."""

    def test_search_returns_results(self) -> None:
        search = SearchTool()
        search.load_corpus()
        result = search.execute({"query": "population Springfield"})
        assert "Springfield" in result
        assert "116,250" in result

    def test_search_relevance(self) -> None:
        search = SearchTool()
        search.load_corpus()
        result = search.execute({"query": "programming language"})
        assert "Python" in result

    def test_search_no_results(self) -> None:
        search = SearchTool()
        search.load_corpus()
        result = search.execute({"query": "xyznonexistent12345"})
        assert "No results" in result

    def test_bm25_index(self) -> None:
        index = BM25Index()
        docs = [
            {"text": "the cat sat on the mat", "id": "1"},
            {"text": "the dog played in the park", "id": "2"},
        ]
        index.add_documents(docs)
        results = index.search("cat mat")
        assert len(results) > 0
        assert results[0][0]["id"] == "1"


class TestFileStore:
    """Tests for the in-memory file store."""

    def test_write_and_read(self) -> None:
        fs = FileStoreTool()
        fs.execute({"action": "write", "key": "test", "value": "hello"})
        result = fs.execute({"action": "read", "key": "test"})
        assert result == "hello"

    def test_list_keys(self) -> None:
        fs = FileStoreTool()
        fs.execute({"action": "write", "key": "a", "value": "1"})
        fs.execute({"action": "write", "key": "b", "value": "2"})
        result = fs.execute({"action": "list"})
        assert "a" in result
        assert "b" in result

    def test_delete(self) -> None:
        fs = FileStoreTool()
        fs.execute({"action": "write", "key": "x", "value": "y"})
        fs.execute({"action": "delete", "key": "x"})
        with pytest.raises(ValueError, match="Key not found"):
            fs.execute({"action": "read", "key": "x"})

    def test_read_nonexistent(self) -> None:
        fs = FileStoreTool()
        with pytest.raises(ValueError, match="Key not found"):
            fs.execute({"action": "read", "key": "nope"})
