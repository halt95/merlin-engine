# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 halt95 (Merlin modifications)
"""Closed integer interval helpers."""


def merge(intervals):
    """Merge overlapping or touching (start, end) intervals into a new sorted list."""
    result = []
    for start, end in sorted(intervals):
        if result and start <= result[-1][1]:
            result[-1] = (result[-1][0], max(result[-1][1], end))
        else:
            result.append((start, end))
    return result


def total_length(intervals):
    """Return the length covered by the intervals: the sum of end - start over the merged list."""
    return sum(end - start for start, end in merge(intervals))
