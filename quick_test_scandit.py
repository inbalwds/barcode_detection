#!/usr/bin/env python3
"""
Quick test script to test Scandit-based L1 barcode replacement on a single image
"""

import os
import sys
import yaml
import cv2
from pathlib import Path

# Add project root to path
sys.path.append('/home/ubuntu/barcode_detection')
sys.path.append('/home/ubuntu/barcode_detection/data')

from data.improved_warehouse_generator import ImprovedWarehouseBarcodeGenerator

def main():
    # Load config
    with open('/home/ubuntu/barcode_detection/config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    class_mapping = {
        'box': 0,
        'empty_shelves': 1,
        'palletes': 2,
        'product_barcodes': 3,
        'shelf': 4,
        'shelf_barcodes': 5
    }

    print("="*70)
    print("QUICK TEST: L1 DataMatrix Replacement")
    print("="*70)
    print()

    # Use a high-res image that we know has L1 DataMatrix barcodes
    test_image = '/home/ubuntu/barcode_detection/hires1_imgs/hires1_301.466689772_176.png'

    # Create a dummy label file
    test_label = '/home/ubuntu/barcode_detection/quick_test_output/dummy_label.txt'
    os.makedirs(os.path.dirname(test_label), exist_ok=True)

    print(f"Test image: {test_image}")
    print(f"Test label: {test_label}")
    print()

    # Check if label exists
    if not os.path.exists(test_label):
        print(f"Warning: Label file not found, creating empty label file")
        os.makedirs(os.path.dirname(test_label), exist_ok=True)
        open(test_label, 'w').close()

    # Initialize generator
    print("Initializing generator with Scandit...")
    generator = ImprovedWarehouseBarcodeGenerator(
        class_mapping=class_mapping,
        scandit_license_key=config['license_key'],
        use_scandit=True
    )
    print()

    # Process single image
    print("Processing image...")
    output_dir = '/home/ubuntu/barcode_detection/quick_test_output'
    os.makedirs(output_dir, exist_ok=True)

    try:
        syn_image, syn_annotations = generator.generate_synthetic_image(
            image_path=test_image,
            annotation_path=test_label,
            num_barcodes_range=(2, 4),
            replace_detected=True,  # Try to replace detected barcodes
            add_random_barcodes=True  # If no barcodes detected, add random ones
        )

        # Save result
        output_image = os.path.join(output_dir, 'test_result.jpg')
        output_label = os.path.join(output_dir, 'test_result.txt')

        cv2.imwrite(output_image, syn_image)
        generator.save_yolo_annotations(syn_annotations, output_label)

        print()
        print("="*70)
        print("✅ SUCCESS!")
        print("="*70)
        print(f"Output saved to:")
        print(f"  - Image: {output_image}")
        print(f"  - Label: {output_label}")
        print()

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0

if __name__ == "__main__":
    exit(main())
