# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 halt95 (Merlin modifications)
"""A tiny shopping cart that keeps money in Decimal."""

from decimal import ROUND_HALF_UP, Decimal


class Cart:
    def __init__(self, items=None):
        self.items = list(items) if items else []

    def add(self, name, unit_price, qty=1):
        """Add qty units of an item; unit_price is a decimal string such as '2.50'."""
        if qty <= 0:
            raise ValueError("qty must be positive")
        self.items.append((name, Decimal(unit_price), qty))

    def subtotal(self):
        return sum((price * qty for _, price, qty in self.items), Decimal("0"))

    def total(self, discount_percent=0):
        """Subtotal less discount_percent percent, rounded half-up to cents."""
        if not 0 <= discount_percent <= 100:
            raise ValueError("discount_percent must be between 0 and 100")
        sub = self.subtotal()
        total = sub - sub * Decimal(str(discount_percent)) / 100
        return total.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
