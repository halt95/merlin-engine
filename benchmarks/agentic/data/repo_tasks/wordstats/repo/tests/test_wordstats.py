# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 halt95 (Merlin modifications)
import unittest

from wordstats import top_words, word_counts


class WordCountsTest(unittest.TestCase):
    def test_case_insensitive(self):
        self.assertEqual(word_counts("The cat the"), {"the": 2, "cat": 1})

    def test_any_whitespace(self):
        self.assertEqual(word_counts("a  b\nc\ta"), {"a": 2, "b": 1, "c": 1})

    def test_punctuation_stripped(self):
        self.assertEqual(word_counts("Hi! hi, HI."), {"hi": 3})

    def test_punctuation_only_tokens_ignored(self):
        self.assertEqual(word_counts("  ... ! "), {})


class TopWordsTest(unittest.TestCase):
    def test_most_frequent_first(self):
        self.assertEqual(top_words("x y y z z z", 2), [("z", 3), ("y", 2)])

    def test_ties_alphabetical(self):
        self.assertEqual(top_words("b a c b a", 2), [("a", 2), ("b", 2)])


if __name__ == "__main__":
    unittest.main()
