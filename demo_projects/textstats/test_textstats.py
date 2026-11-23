from textstats import average_word_length, top_words, word_count


def test_word_count_simple():
    assert word_count("a b a") == {"a": 2, "b": 1}


def test_word_count_ignores_case_and_punctuation():
    assert word_count("Hello, hello world!") == {"hello": 2, "world": 1}


def test_word_count_handles_multiple_spaces_and_newlines():
    assert word_count("one  two\nthree") == {"one": 1, "two": 1, "three": 1}


def test_word_count_empty():
    assert word_count("") == {}


def test_average_word_length():
    assert average_word_length("ab abcd") == 3.0


def test_average_word_length_empty():
    assert average_word_length("") == 0.0


def test_top_words():
    assert top_words("b a b c a b", 2) == [("b", 3), ("a", 2)]
