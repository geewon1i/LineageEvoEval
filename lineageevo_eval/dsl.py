"""LineageEvo-compatible AlphaPROBE-style expression support."""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass


TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|-?\d+(?:\.\d+)?|[()+\-*/,]")


@dataclass(frozen=True)
class FactorExpression:
    raw: str

    @property
    def tokens(self) -> list[str]:
        return TOKEN_RE.findall(self.raw)

    @property
    def length(self) -> int:
        return len(self.tokens)


@dataclass(frozen=True)
class FactorDSL:
    features: tuple[str, ...] = ("$open", "$high", "$low", "$close", "$vwap", "$volume")
    rolling_constants: tuple[int, ...] = (1, 3, 5, 10, 20, 30, 60)
    arithmetic_constants: tuple[float, ...] = (0.0001, 0.001, 0.01, 0.0, 1.0, 2.0)


DEFAULT_DSL = FactorDSL()


class QlibExpressionError(ValueError):
    """Raised when an expression cannot be converted to a Qlib expression."""


class QlibExpressionNormalizer:
    """Convert LineageEvo AlphaPROBE-style expressions to Qlib expressions."""

    def __init__(self, dsl: FactorDSL | None = None) -> None:
        self.dsl = dsl or DEFAULT_DSL

    def normalize(self, expression: FactorExpression | str) -> str:
        raw = expression.raw if isinstance(expression, FactorExpression) else expression
        tree = ast.parse(self._preprocess(raw), mode="eval")
        return self._convert(tree.body)

    @staticmethod
    def _preprocess(raw: str) -> str:
        return re.sub(r"\$([A-Za-z_][A-Za-z0-9_]*)", r"\1", raw.strip())

    def _convert(self, node: ast.AST) -> str:
        if isinstance(node, ast.Name):
            feature = f"${node.id.lower()}"
            if feature not in self.dsl.features:
                raise QlibExpressionError(f"unknown feature: {node.id}")
            return feature
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return self._constant(node.value)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            if isinstance(node.operand, ast.Constant) and isinstance(node.operand.value, (int, float)):
                return self._constant(-node.operand.value)
            return f"(-{self._convert(node.operand)})"
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            args = [self._convert(arg) for arg in node.args]
            return self._call(node.func.id, args)
        raise QlibExpressionError(f"unsupported syntax: {type(node).__name__}")

    def _constant(self, value: int | float) -> str:
        if value in self.dsl.rolling_constants or float(value) in self.dsl.arithmetic_constants:
            return str(value)
        raise QlibExpressionError(f"constant not allowed: {value}")

    def _call(self, func_name: str, args: list[str]) -> str:
        func = func_name.lower()
        if func in {"abs", "log", "sign", "rank", "slog1p"}:
            self._expect(func_name, args, 1)
            return self._unary(func, args[0])
        if func in {"add", "sub", "mul", "div", "pow", "greater", "less", "getgreater", "getless"}:
            self._expect(func_name, args, 2)
            return self._binary(func, args[0], args[1])
        if func == "ref":
            self._expect(func_name, args, 2)
            self._require_rolling_constant(args[1], func_name)
            return f"Ref({args[0]}, {args[1]})"
        if func.startswith("ts"):
            return self._timeseries(func, func_name, args)
        raise QlibExpressionError(f"unknown operator: {func_name}")

    @staticmethod
    def _expect(func_name: str, args: list[str], count: int) -> None:
        if len(args) != count:
            raise QlibExpressionError(f"{func_name} expects {count} arguments")

    def _require_rolling_constant(self, value: str, func_name: str) -> None:
        try:
            parsed = int(float(value))
        except ValueError as exc:
            raise QlibExpressionError(f"{func_name} window must be a constant") from exc
        if parsed not in self.dsl.rolling_constants:
            raise QlibExpressionError(f"{func_name} window constant not allowed: {value}")

    @staticmethod
    def _unary(func: str, x: str) -> str:
        if func == "abs":
            return f"Abs({x})"
        if func == "log":
            return f"Log({x})"
        if func == "sign":
            return f"Sign({x})"
        if func == "rank":
            return f"Rank({x}, 5)"
        if func == "slog1p":
            return f"Sign({x}) * Log(Abs({x}) + 1)"
        raise QlibExpressionError(f"unknown unary operator: {func}")

    @staticmethod
    def _binary(func: str, x: str, y: str) -> str:
        if func == "add":
            return f"({x} + {y})"
        if func == "sub":
            return f"({x} - {y})"
        if func == "mul":
            return f"({x} * {y})"
        if func == "div":
            return f"({x} / {y})"
        if func == "pow":
            return f"Power({x}, {y})"
        if func == "greater":
            return f"Gt({x}, {y})"
        if func == "less":
            return f"Lt({x}, {y})"
        if func == "getgreater":
            return f"Greater({x}, {y})"
        if func == "getless":
            return f"Less({x}, {y})"
        raise QlibExpressionError(f"unknown binary operator: {func}")

    def _timeseries(self, func: str, func_name: str, args: list[str]) -> str:
        three_arg = {"tscov": "Cov", "tscorr": "Corr"}
        two_arg = {
            "tsmean": "Mean",
            "tssum": "Sum",
            "tsstd": "Std",
            "tsmin": "Min",
            "tsmax": "Max",
            "tsvar": "Var",
            "tsskew": "Skew",
            "tskurt": "Kurt",
            "tsmed": "Med",
            "tsmad": "Mad",
            "tsrank": "Rank",
            "tsdelta": "Delta",
            "tsema": "EMA",
            "tswma": "WMA",
            "tsir": "Mean",
        }
        if func in three_arg:
            self._expect(func_name, args, 3)
            self._require_rolling_constant(args[2], func_name)
            return f"{three_arg[func]}({args[0]}, {args[1]}, {args[2]})"
        self._expect(func_name, args, 2)
        self._require_rolling_constant(args[1], func_name)
        x, d = args
        if func == "tsminmaxdiff":
            return f"(Max({x}, {d}) - Min({x}, {d}))"
        if func == "tsmaxdiff":
            return f"({x} - Max({x}, {d}))"
        if func == "tsmindiff":
            return f"({x} - Min({x}, {d}))"
        if func == "tsratio":
            return f"({x} / Ref({x}, {d}))"
        if func == "tspctchange":
            return f"({x} / Ref({x}, {d}) - 1)"
        if func == "tsir":
            return f"(Mean({x}, {d}) / Std({x}, {d}))"
        if func in two_arg:
            return f"{two_arg[func]}({x}, {d})"
        raise QlibExpressionError(f"unknown operator: {func_name}")


def validate_expression_length(expression: FactorExpression, max_length: int) -> None:
    if expression.length > max_length:
        raise QlibExpressionError(f"factor length {expression.length} exceeds {max_length}")
