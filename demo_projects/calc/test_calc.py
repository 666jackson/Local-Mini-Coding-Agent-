import pytest

from calc import CalcError, evaluate


def test_addition():
    assert evaluate("1 + 2") == 3


def test_subtraction_left_to_right():
    assert evaluate("10 - 3 - 2") == 5


def test_precedence_mul_before_add():
    assert evaluate("2 + 3 * 4") == 14


def test_precedence_div_before_sub():
    assert evaluate("10 - 6 / 2") == 7


def test_division():
    assert evaluate("7 / 2") == 3.5


def test_division_by_zero_raises_calc_error():
    with pytest.raises(CalcError, match="division by zero"):
        evaluate("1 / 0")


def test_empty_expression():
    with pytest.raises(CalcError):
        evaluate("   ")


def test_invalid_character():
    with pytest.raises(CalcError):
        evaluate("2 + x")
