# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 halt95 (Merlin modifications)
import unittest
from decimal import Decimal

from cart import Cart


class CartTest(unittest.TestCase):
    def test_carts_do_not_share_items(self):
        a = Cart()
        b = Cart()
        a.add("pen", "1.00")
        self.assertEqual(b.items, [])

    def test_empty_cart_total(self):
        self.assertEqual(Cart().total(), Decimal("0.00"))

    def test_half_up_rounding(self):
        c = Cart()
        c.add("washer", "0.125")
        self.assertEqual(c.total(), Decimal("0.13"))

    def test_discount(self):
        c = Cart()
        c.add("lamp", "19.99")
        self.assertEqual(c.total(10), Decimal("17.99"))

    def test_quantity(self):
        c = Cart()
        c.add("cup", "2.50", 3)
        self.assertEqual(c.total(), Decimal("7.50"))

    def test_discount_out_of_range(self):
        c = Cart()
        c.add("cup", "2.50")
        with self.assertRaises(ValueError):
            c.total(150)
        with self.assertRaises(ValueError):
            c.total(-5)

    def test_zero_quantity_rejected(self):
        with self.assertRaises(ValueError):
            Cart().add("cup", "2.50", 0)


if __name__ == "__main__":
    unittest.main()
