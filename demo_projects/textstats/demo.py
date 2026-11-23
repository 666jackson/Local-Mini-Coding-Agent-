"""示範腳本：用幾個例子呼叫 textstats 的函式，印出「結果」和「正確答案」。

mini-agent 會在修改前、修改後各跑一次，並排顯示差別。每行格式：`例子 => 結果 => 正確答案`。
"""

from textstats import average_word_length, top_words, word_count


def show(label: str, fn, arg, expected: str) -> None:
    """執行一個例子；就算程式當掉也印出來。"""
    try:
        result = str(fn(*arg))
    except Exception as e:  # noqa: BLE001
        result = f"{type(e).__name__}: {e}"
    print(f"{label} => {result} => {expected}")


show('word_count("Hello, hello world!")', word_count, ("Hello, hello world!",), "{'hello': 2, 'world': 1}")
show('top_words("The cat. The dog! the end", 1)', top_words, ("The cat. The dog! the end", 1), "[('the', 3)]")
show('average_word_length("")', average_word_length, ("",), "0.0")
