"""示範腳本：用幾個例子呼叫 evaluate，印出「結果」和「正確答案」。

mini-agent 會在修改前、修改後各跑一次，並排顯示差別。每行格式：`例子 => 結果 => 正確答案`。
（錯誤只比對例外類別，訊息文字不比對。）
"""

from calc import CalcError, evaluate

EXAMPLES = [("2 + 3 * 4", "14.0"), ("10 - 6 / 2", "7.0"), ("8 / 2 * 3", "12.0"), ("1 / 0", "CalcError")]

for expr, expected in EXAMPLES:
    try:
        result = str(evaluate(expr))
    except CalcError as e:
        result = f"CalcError: {e}"
    except Exception as e:  # noqa: BLE001 - 示範用：把當掉也印出來
        result = f"{type(e).__name__}: {e}"
    print(f"evaluate({expr!r}) => {result} => {expected}")
