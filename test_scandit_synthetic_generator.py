#!/usr/bin/env python3
"""
Test script for Scandit-integrated synthetic barcode generator
This script demonstrates how to:
1. Use Scandit to detect L1-prefixed DataMatrix codes in images
2. Replace detected L1 DataMatrix codes with newly generated ones
3. Add additional synthetic barcodes to the images
"""

import os
import sys
import yaml
from pathlib import Path

# Add project root to path
sys.path.append('/home/ubuntu/barcode_detection')
sys.path.append('/home/ubuntu/barcode_detection/data')

from data.improved_warehouse_generator import ImprovedWarehouseBarcodeGenerator

def load_config(config_path='/home/ubuntu/barcode_detection/config.yaml'):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def main():
    print("="*70)
    print("SCANDIT-INTEGRATED SYNTHETIC BARCODE GENERATOR TEST")
    print("="*70)
    print()

    # Load configuration
    config = load_config()

    # Configuration
    class_mapping = {
        'box': 0,
        'empty_shelves': 1,
        'palletes': 2,
        'product_barcodes': 3,
        'shelf': 4,
        'shelf_barcodes': 5
    }

    # Test paths - adjust these as needed
    INPUT_IMAGES_DIR = config.get('INPUT_IMAGES_DIR',
                                   '/home/ubuntu/barcode_detection/rami levy.v16i.yolov12/train/images')
    INPUT_LABELS_DIR = config.get('TEST_LABELS_DIR',
                                   '/home/ubuntu/barcode_detection/test_improved/detected_labels')

    OUTPUT_IMAGES_DIR = '/home/ubuntu/barcode_detection/test_output/scandit_synthetic_images'
    OUTPUT_LABELS_DIR = '/home/ubuntu/barcode_detection/test_output/scandit_synthetic_labels'

    SCANDIT_LICENSE_KEY = config.get('license_key')

    print(f"Configuration:")
    print(f"  - Input images: {INPUT_IMAGES_DIR}")
    print(f"  - Input labels: {INPUT_LABELS_DIR}")
    print(f"  - Output images: {OUTPUT_IMAGES_DIR}")
    print(f"  - Output labels: {OUTPUT_LABELS_DIR}")
    print(f"  - Scandit license: {'✓ Loaded' if SCANDIT_LICENSE_KEY else '✗ Not found'}")
    print()

    if not SCANDIT_LICENSE_KEY:
        print("ERROR: No Scandit license key found in config.yaml")
        return 1

    # Check if input directories exist
    if not os.path.exists(INPUT_IMAGES_DIR):
        print(f"ERROR: Input images directory not found: {INPUT_IMAGES_DIR}")
        return 1

    if not os.path.exists(INPUT_LABELS_DIR):
        print(f"ERROR: Input labels directory not found: {INPUT_LABELS_DIR}")
        return 1

    # Initialize generator with Scandit support
    print("Initializing generator with Scandit detection...")
    generator = ImprovedWarehouseBarcodeGenerator(
        class_mapping=class_mapping,
        scandit_license_key=SCANDIT_LICENSE_KEY,
        use_scandit=True
    )
    print()

    # Process dataset
    print("Processing dataset...")
    print("  - Strategy:")
    print("    1. Detect L1-prefixed DataMatrix codes using Scandit")
    print("    2. Replace detected L1 codes with newly generated L1 codes")
    print("    3. Add additional synthetic barcodes (Code128 on shelves, DataMatrix on pallets)")
    print()

    generator.process_dataset(
        images_dir=INPUT_IMAGES_DIR,
        labels_dir=INPUT_LABELS_DIR,
        output_images_dir=OUTPUT_IMAGES_DIR,
        output_labels_dir=OUTPUT_LABELS_DIR,
        num_augmentations_per_image=2,  # Create 2 versions of each image
        replace_detected_l1=True  # Enable L1 replacement
    )

    print()
    print("="*70)
    print("✅ COMPLETE!")
    print("="*70)
    print()
    print(f"Results saved to:")
    print(f"  - Images: {OUTPUT_IMAGES_DIR}")
    print(f"  - Labels: {OUTPUT_LABELS_DIR}")
    print()
    print("What happened:")
    print("  1. Scandit detected all DataMatrix codes starting with 'L1'")
    print("  2. Each detected L1 code was replaced with a new generated L1 code")
    print("  3. Additional synthetic barcodes were added to the images")
    print()
    print("Next steps:")
    print("  - Check the output images to verify barcode replacement")
    print("  - Use the synthetic dataset for training your YOLO model")
    print()

if __name__ == "__main__":
    exit(main() or 0)
