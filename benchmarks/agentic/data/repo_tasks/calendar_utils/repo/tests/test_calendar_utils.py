# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 halt95 (Merlin modifications)
import unittest

from calendar_utils import day_of_year, days_in_month, is_leap


class LeapTest(unittest.TestCase):
    def test_leap_years(self):
        self.assertTrue(is_leap(2024))
        self.assertTrue(is_leap(2000))
        self.assertFalse(is_leap(2023))
        self.assertFalse(is_leap(1900))


class MonthTest(unittest.TestCase):
    def test_month_lengths(self):
        self.assertEqual(days_in_month(2023, 1), 31)
        self.assertEqual(days_in_month(2023, 4), 30)
        self.assertEqual(days_in_month(2023, 12), 31)

    def test_february(self):
        self.assertEqual(days_in_month(2024, 2), 29)
        self.assertEqual(days_in_month(1900, 2), 28)

    def test_bad_month(self):
        for month in (0, 13):
            with self.assertRaises(ValueError):
                days_in_month(2023, month)


class DayOfYearTest(unittest.TestCase):
    def test_day_of_year(self):
        self.assertEqual(day_of_year(2024, 3, 1), 61)
        self.assertEqual(day_of_year(2023, 12, 31), 365)
        self.assertEqual(day_of_year(2023, 1, 1), 1)


if __name__ == "__main__":
    unittest.main()
