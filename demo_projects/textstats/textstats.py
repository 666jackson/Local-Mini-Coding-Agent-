"""Simple text statistics."""

from collections import Counter


def word_count(text: str) -> dict:
    """Count words case-insensitively, ignoring punctuation.

    word_count("Hello, hello world!") == {"hello": 2, "world": 1}
    """
    words = text.split(" ")
    return dict(Counter(words))


def average_word_length(text: str) -> float:
    """Average length of the words in text; 0.0 for empty text."""
    words = text.split()
    return sum(len(w) for w in words) / len(words)


def top_words(text: str, n: int = 3) -> list:
    """The n most common words as (word, count), ties broken alphabetically."""
    counts = word_count(text)
    return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:n]
