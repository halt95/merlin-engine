# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 halt95 (Merlin modifications)
"""Gregorian calendar helpers."""

_DAYS = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]


def is_leap(year):
    return year % 4 == 0


def days_in_month(year, month):
    """Number of days in month (1 = January) of year."""
    if month == 2 and is_leap(year):
        return 29
    return _DAYS[month]


def day_of_year(year, month, day):
    """1-based ordinal of the date within its year."""
    return sum(days_in_month(year, m) for m in range(1, month)) + day
