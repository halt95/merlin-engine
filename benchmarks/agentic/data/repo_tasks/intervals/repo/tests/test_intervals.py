# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 halt95 (Merlin modifications)
import unittest

from intervals import merge, total_length


class MergeTest(unittest.TestCase):
    def test_empty(self):
        self.assertEqual(merge([]), [])

    def test_unsorted_input(self):
        self.assertEqual(merge([(5, 7), (1, 3)]), [(1, 3), (5, 7)])

    def test_touching_intervals_merge(self):
        self.assertEqual(merge([(1, 3), (3, 5)]), [(1, 5)])

    def test_contained_interval(self):
        self.assertEqual(merge([(1, 10), (2, 3)]), [(1, 10)])

    def test_input_not_modified(self):
        data = [(4, 6), (1, 2)]
        merge(data)
        self.assertEqual(data, [(4, 6), (1, 2)])


class TotalLengthTest(unittest.TestCase):
    def test_overlaps_counted_once(self):
        self.assertEqual(total_length([(8, 9), (1, 4), (2, 6)]), 6)


if __name__ == "__main__":
    unittest.main()
