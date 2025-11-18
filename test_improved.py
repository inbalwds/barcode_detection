"""
Quick test of improved pipeline with realistic barcode sizes
"""

import os
from pathlib import Path
import shutil
import yaml

from detect_and_generate_dataset import PalletShelfDetector
from data.improved_warehouse_generator import ImprovedWarehouseBarcodeGenerator
from scandit.barcode_detector import ScanditDecoder


def main():
    """Test improved pipeline on 5 images"""
    with open('config.yaml', 'r') as file:
        data = yaml.safe_load(file)

    MODEL_PATH = data['MODEL_PATH']
    INPUT_IMAGES_DIR = data['INPUT_IMAGES_DIR']

    # Test output
    TEST_LABELS_DIR = data['TEST_LABELS_DIR']
    TEST_VIS_DIR = data['TEST_VIS_DIR']
    TEST_SYNTHETIC_IMAGES = data['TEST_SYNTHETIC_IMAGES']
    TEST_SYNTHETIC_LABELS = data['TEST_SYNTHETIC_LABELS']

    CLASS_NAMES = data['CLASS_NAMES']
    license_key = data['license_key']

    print("="*70)
    print("QUICK TEST - Improved Pipeline (5 images)")
    print("="*70)
    print("\nFeatures:")
    print("  ✓ Code128 on shelves with realistic shelf_barcode sizes")
    print("  ✓ DataMatrix on pallets with realistic product_barcode sizes")
    print("="*70)

    # Get first 5 images
    image_files = list(Path(INPUT_IMAGES_DIR).glob('*.jpg'))[:5]

    if not image_files:
        print("No images found!")
        return

    # Temp directory
    temp_images_dir = '/home/ubuntu/barcode_detection/test_improved/temp_images'
    os.makedirs(temp_images_dir, exist_ok=True)

    # Copy images
    for img_path in image_files:
        shutil.copy(str(img_path), temp_images_dir)

    print(f"\nStep 1: Detecting pallets/shelves...")

    # Detect
    detector = PalletShelfDetector(
        model_path=MODEL_PATH,
        class_names=CLASS_NAMES
    )

    stats = detector.process_directory(
        input_images_dir=temp_images_dir,
        output_labels_dir=TEST_LABELS_DIR,
        conf_threshold=0.25,
        visualize=True,
        output_vis_dir=TEST_VIS_DIR
    )

    scandit_detector = ScanditDecoder(license_key=license_key)
    scandit_detector.

    # run barcode detector on the image 

    if stats['with_detections'] == 0:
        print("No detections found!")
        return

    print(f"\nStep 2: Generating synthetic barcodes with realistic sizes...")
    print(f"  - Code128: ~0.073 × 0.046 (normalized, on shelves)")
    print(f"  - DataMatrix: ~0.063 × 0.081 (normalized, on pallets)")

    # Generate with improved generator
    class_mapping = {
        'box': 0,
        'empty_shelves': 1,
        'palletes': 2,
        'product_barcodes': 3,
        'shelf': 4,
        'shelf_barcodes': 5
    }

    # Real size ranges
    shelf_barcode_size_range = (
        (0.0232, 0.1481),  # width
        (0.0235, 0.0904)   # height
    )

    product_barcode_size_range = (
        (0.0292, 0.1204),  # width
        (0.0445, 0.1520)   # height
    )

    generator = ImprovedWarehouseBarcodeGenerator(
        class_mapping=class_mapping,
        shelf_barcode_size_range=shelf_barcode_size_range,
        product_barcode_size_range=product_barcode_size_range
    )

    generator.process_dataset(
        images_dir=temp_images_dir,
        labels_dir=TEST_LABELS_DIR,
        output_images_dir=TEST_SYNTHETIC_IMAGES,
        output_labels_dir=TEST_SYNTHETIC_LABELS,
        num_augmentations_per_image=2
    )

    print("\n" + "="*70)
    print("TEST COMPLETE!")
    print("="*70)
    print(f"\nResults:")
    print(f"  - Detection visualizations: {TEST_VIS_DIR}")
    print(f"  - Synthetic images: {TEST_SYNTHETIC_IMAGES}")
    print(f"  - Synthetic labels: {TEST_SYNTHETIC_LABELS}")
    print(f"\nCheck the synthetic images to see:")
    print(f"  - Code128 barcodes (horizontal) on SHELVES")
    print(f"  - DataMatrix barcodes (square) on PALLETS")
    print(f"  - Both with realistic sizes matching your training data!")


if __name__ == "__main__":
    main()
