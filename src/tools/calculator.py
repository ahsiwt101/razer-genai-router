"""calculator tool
seperate so that user cant reach eval 

- parse text into tree 
- all names/attribute access/imports/strings are rejected 

limits so "9 ** 9 ** 9" can't freeze the machine computing a number with millions of digits.
"""
import ast
import math
import operator

from .base import ToolResult

_BIN_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod, ast.Pow: operator.pow,
}
_UNARY_OPS = {ast.USub: operator.neg, ast.UAdd: operator.pos}
_FUNCS = {
    "sqrt": math.sqrt, "cbrt": lambda x: math.copysign(abs(x) ** (1 / 3), x),
    "log": math.log10, "ln": math.log, "abs": abs, "round": round,
    "factorial": lambda x: math.factorial(int(x)),
}
MAX_EXPONENT = 1000      # 2 ** 1000 is fine; 9 ** 9 ** 9 is not
MAX_FACTORIAL = 500
MAX_RESULT_BITS = 4000   # that is roughly 1200 digits


class UnsafeExpression(ValueError):
    pass


def _eval(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        left, right = _eval(node.left), _eval(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > MAX_EXPONENT:
            raise UnsafeExpression(f"exponent {right} is too large")
        result = _BIN_OPS[type(node.op)](left, right)
        if isinstance(result, int) and result.bit_length() > MAX_RESULT_BITS:
            raise UnsafeExpression("result is too large")
        return result
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_eval(node.operand))
    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _FUNCS
            and len(node.args) == 1 and not node.keywords):
        arg = _eval(node.args[0])
        if node.func.id == "factorial" and arg > MAX_FACTORIAL:
            raise UnsafeExpression(f"factorial of {arg} is too large")
        return _FUNCS[node.func.id](arg)
    raise UnsafeExpression(f"'{type(node).__name__}' is not allowed in a calculation")


def evaluate(expression: str) -> float:
    # safely evaluate an arithmetic expression. 
    # raises UnsafeExpression / ZeroDivisionError / SyntaxError
    tree = ast.parse(expression.replace("^", "**"), mode="eval")  
    return _eval(tree.body)


def _format(value: float) -> str:
    if isinstance(value, float) and value.is_integer() and abs(value) < 1e15:
        value = int(value)
    if isinstance(value, float):
        return f"{value:.6g}" if abs(value) < 1e15 else f"{value:.6e}"
    if abs(value) < 10**15:
        return f"{value:,}"
    # huge integers too big to convert to float -> build scientific notation by hand
    digits = str(abs(value))
    return f"{'-' if value < 0 else ''}{digits[0]}.{digits[1:7]}e+{len(digits) - 1}"


def run(expression: str) -> ToolResult:
    try:
        value = evaluate(expression)
    except ZeroDivisionError:
        return ToolResult("calculator", ok=False, error="division by zero")
    except (UnsafeExpression, SyntaxError, ValueError, OverflowError, TypeError) as e:
        return ToolResult("calculator", ok=False, error=f"could not calculate '{expression}': {e}")
    if isinstance(value, float) and not math.isfinite(value):  # e.g. 1e300 * 1e300 -> inf
        return ToolResult("calculator", ok=False, error="result is too large (overflow)")
    return ToolResult("calculator", ok=True, output=f"{expression} = {_format(value)}",
                      data={"expression": expression, "value": value})
