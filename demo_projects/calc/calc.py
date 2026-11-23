"""A tiny arithmetic calculator: evaluate("2 + 3 * 4") -> 14."""

import re

TOKEN_RE = re.compile(r"\s*(\d+\.?\d*|[+\-*/])")


class CalcError(Exception):
    """Raised for invalid expressions or math errors."""


def tokenize(expr: str) -> list:
    """Split an expression into numbers and operators."""
    tokens = []
    pos = 0
    expr = expr.strip()
    while pos < len(expr):
        m = TOKEN_RE.match(expr, pos)
        if not m:
            raise CalcError(f"unexpected character at {pos}: {expr[pos]!r}")
        tok = m.group(1)
        tokens.append(float(tok) if tok[0].isdigit() else tok)
        pos = m.end()
    return tokens


def apply(op: str, a: float, b: float) -> float:
    """Apply a binary operator."""
    if op == "+":
        return a + b
    if op == "-":
        return a - b
    if op == "*":
        return a * b
    if op == "/":
        return a / b
    raise CalcError(f"unknown operator {op!r}")


def evaluate(expr: str) -> float:
    """Evaluate an expression with + - * / on numbers."""
    tokens = tokenize(expr)
    if not tokens:
        raise CalcError("empty expression")
    if len(tokens) % 2 == 0:
        raise CalcError("incomplete expression")
    result = tokens[0]
    for i in range(1, len(tokens), 2):
        result = apply(tokens[i], result, tokens[i + 1])
    return result
