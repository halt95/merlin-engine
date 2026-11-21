# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 halt95 (Merlin modifications)
"""Small word statistics helpers."""

PUNCTUATION = ".,;:!?"


def word_counts(text):
    """Return a dict mapping each word in text to how many times it occurs."""
    counts = {}
    for word in text.split():
        w = word.strip(PUNCTUATION).lower()
        if w:
            counts[w] = counts.get(w, 0) + 1
    return counts


def top_words(text, n):
    """Return the n most frequent (word, count) pairs, most frequent first."""
    counts = word_counts(text)
    return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:n]
