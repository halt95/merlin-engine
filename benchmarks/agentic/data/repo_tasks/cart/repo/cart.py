# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 halt95 (Merlin modifications)
"""A tiny shopping cart that keeps money in Decimal."""

from decimal import Decimal


class Cart:
    def __init__(self, items=[]):
        self.items = items

    def add(self, name, unit_price, qty=1):
        """Add qty units of an item; unit_price is a decimal string such as '2.50'."""
        if qty <= 0:
            raise ValueError("qty must be positive")
        self.items.append((name, Decimal(unit_price), qty))

    def subtotal(self):
        return sum(price * qty for _, price, qty in self.items)

    def total(self, discount_percent=0):
        """Subtotal less discount_percent percent, rounded half-up to cents."""
        sub = self.subtotal()
        total = sub - sub * discount_percent / 100
        return total.quantize(Decimal("0.01"))
