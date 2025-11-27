#!/usr/bin/env python3
"""
Test script to verify Rami Levi barcode pattern generation
"""

import sys
import re
sys.path.append('/home/ubuntu/barcode_detection')
sys.path.append('/home/ubuntu/barcode_detection/data')

from data.improved_warehouse_generator import ImprovedWarehouseBarcodeGenerator

def test_barcode_patterns():
    """Test that generated barcodes match the expected patterns"""

    # Initialize generator
    class_mapping = {
        'box': 0,
        'empty_shelves': 1,
        'palletes': 2,
        'product_barcodes': 3,
        'shelf': 4,
        'shelf_barcodes': 5
    }

    generator = ImprovedWarehouseBarcodeGenerator(
        class_mapping=class_mapping,
        use_scandit=False  # Don't need Scandit for this test
    )

    # Expected patterns
    product_pattern = r'^L1(0{9}\d{9})$'  # L1 + 9 zeros + 9 digits
    shelf_pattern = r'^\d{4}1\d{2}$'  # 4 digits + "1" + 2 digits

    print("="*80)
    print("RAMI LEVI BARCODE PATTERN TEST")
    print("="*80)
    print()

    # Test Product Barcodes (DataMatrix)
    print("Testing Product Barcodes (DataMatrix):")
    print(f"Expected pattern: {product_pattern}")
    print(f"Example: L1000000000123456789")
    print()

    all_valid = True
    for i in range(10):
        barcode = generator.generate_random_text('datamatrix')
        matches = re.match(product_pattern, barcode)
        status = "✓" if matches else "✗"
        print(f"  {status} Generated: {barcode} {'(VALID)' if matches else '(INVALID)'}")
        if not matches:
            all_valid = False

    print()
    print(f"Product Barcodes: {'✓ ALL VALID' if all_valid else '✗ SOME INVALID'}")
    print()

    # Test Shelf Barcodes (Code128)
    print("Testing Shelf Barcodes (Code128):")
    print(f"Expected pattern: {shelf_pattern}")
    print(f"Example: 2023156")
    print()

    all_valid = True
    for i in range(10):
        barcode = generator.generate_random_text('code128')
        matches = re.match(shelf_pattern, barcode)
        status = "✓" if matches else "✗"
        print(f"  {status} Generated: {barcode} {'(VALID)' if matches else '(INVALID)'}")
        if not matches:
            all_valid = False

    print()
    print(f"Shelf Barcodes: {'✓ ALL VALID' if all_valid else '✗ SOME INVALID'}")
    print()

    print("="*80)
    print("Pattern Breakdown:")
    print("="*80)
    print()
    print("Product Barcode (DataMatrix):")
    print("  Format: L1 + 000000000 + XXXXXXXXX")
    print("          ^    9 zeros    9 digits")
    print("  Total length: 2 + 9 + 9 = 20 characters")
    print()
    print("Shelf Barcode (Code128):")
    print("  Format: XXXX + 1 + XX")
    print("          4 digits  2 digits")
    print("  Total length: 7 digits")
    print("  The '1' is always at position 5 (zero-indexed position 4)")
    print()

if __name__ == "__main__":
    test_barcode_patterns()
