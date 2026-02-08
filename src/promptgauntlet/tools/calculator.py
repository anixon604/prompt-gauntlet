"""Safe calculator tool using AST-based expression evaluation."""

from __future__ import annotations

import ast
import math
import operator
from typing import Any

from promptgauntlet.tools.base import Tool

# Safe operators
_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

# Safe math functions
_FUNCTIONS: dict[str, Any] = {
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "sqrt": math.sqrt,
    "log": math.log,
    "log10": math.log10,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "pi": math.pi,
    "e": math.e,
}


def _safe_eval(node: ast.AST) -> float:
    """Recursively evaluate an AST node safely."""
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    elif isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return float(node.value)
        raise ValueError(f"Unsupported constant type: {type(node.value)}")
    elif isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in _OPERATORS:
            raise ValueError(f"Unsupported operator: {op_type.__name__}")
        left = _safe_eval(node.left)
        right = _safe_eval(node.right)
        return float(_OPERATORS[op_type](left, right))
    elif isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type not in _OPERATORS:
            raise ValueError(f"Unsupported unary operator: {op_type.__name__}")
        operand = _safe_eval(node.operand)
        return float(_OPERATORS[op_type](operand))
    elif isinstance(node, ast.Call):
        if isinstance(node.func, ast.Name) and node.func.id in _FUNCTIONS:
            args = [_safe_eval(a) for a in node.args]
            return float(_FUNCTIONS[node.func.id](*args))
        raise ValueError("Unsupported function call")
    elif isinstance(node, ast.Name):
        if node.id in _FUNCTIONS:
            val = _FUNCTIONS[node.id]
            if isinstance(val, (int, float)):
                return float(val)
        raise ValueError(f"Unknown variable: {node.id}")
    else:
        raise ValueError(f"Unsupported expression type: {type(node).__name__}")


class CalculatorTool(Tool):
    """Safe math expression evaluator.

    Supports basic arithmetic, common math functions, and constants.
    Does NOT use eval() -- parses AST for safety.
    """

    @property
    def name(self) -> str:
        return "calculator"

    @property
    def description(self) -> str:
        return (
            "Evaluate a mathematical expression. Supports: +, -, *, /, //, %, ** "
            "and functions: abs, round, min, max, sqrt, log, log10, sin, cos, tan. "
            "Constants: pi, e."
        )

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Mathematical expression to evaluate, e.g. '2 + 3 * 4'",
                }
            },
            "required": ["expression"],
        }

    def execute(self, arguments: dict[str, Any]) -> str:
        """Evaluate a math expression safely."""
        expr = arguments.get("expression", "")
        if not expr or not isinstance(expr, str):
            raise ValueError("Missing or invalid 'expression' argument")

        # Length limit for safety
        if len(expr) > 500:
            raise ValueError("Expression too long (max 500 characters)")

        try:
            tree = ast.parse(expr, mode="eval")
            result = _safe_eval(tree)
            # Format nicely
            if result == int(result):
                return str(int(result))
            return f"{result:.6f}".rstrip("0").rstrip(".")
        except (SyntaxError, ValueError) as e:
            raise ValueError(f"Invalid expression: {e}") from e
        except ZeroDivisionError as e:
            raise ValueError("Division by zero") from e
