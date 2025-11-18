"""
Improved Pipeline with realistic barcode sizes:
- Code128 on shelves (sized like real shelf_barcodes)
- DataMatrix on pallets (sized like real product_barcodes)
"""

import os
import sys
from pathlib import Path

# Import detector and improved generator
from detect_and_generate_dataset import PalletShelfDetector
from data.improved_warehouse_generator import ImprovedWarehouseBarcodeGenerator


def main():
    """
    Complete improved pipeline
    """

    # Configuration
    MODEL_PATH = '/home/ubuntu/barcode_detection/models/rami_levi_weights_with_first_floor.pt'
    INPUT_IMAGES_DIR = '/home/ubuntu/barcode_detection/rami levy.v16i.yolov12/train/images'

    # Temporary directory for detections
    TEMP_LABELS_DIR = '/home/ubuntu/barcode_detection/temp_detected_labels'

    # Output directories for synthetic dataset
    OUTPUT_IMAGES_DIR = '/home/ubuntu/barcode_detection/synthetic_dataset_v2/images'
    OUTPUT_LABELS_DIR = '/home/ubuntu/barcode_detection/synthetic_dataset_v2/labels'

    # Class names
    CLASS_NAMES = ['box', 'empty_shelves', 'palletes', 'product_barcodes', 'shelf', 'shelf_barcodes']

    print("="*70)
    print("IMPROVED WAREHOUSE SYNTHETIC BARCODE DATASET GENERATOR")
    print("="*70)
    print("\nFeatures:")
    print("  - Code128 barcodes on SHELVES (realistic shelf_barcode sizes)")
    print("  - DataMatrix barcodes on PALLETS (realistic product_barcode sizes)")
    print("="*70)

    print(f"\nStep 1: Detecting pallets and shelves...")
    print(f"  Model: {MODEL_PATH}")
    print(f"  Input: {INPUT_IMAGES_DIR}")

    # Step 1: Detect pallets and shelves
    detector = PalletShelfDetector(
        model_path=MODEL_PATH,
        class_names=CLASS_NAMES
    )

    stats = detector.process_directory(
        input_images_dir=INPUT_IMAGES_DIR,
        output_labels_dir=TEMP_LABELS_DIR,
        conf_threshold=0.25,
        visualize=False  # Skip visualization for speed
    )

    if stats['with_detections'] == 0:
        print("\n⚠ Warning: No detections found!")
        return

    print(f"\nStep 2: Generating synthetic barcodes with realistic sizes...")
    print(f"  - Code128 size: 0.023-0.148 × 0.024-0.090 (normalized)")
    print(f"  - DataMatrix size: 0.029-0.120 × 0.045-0.152 (normalized)")
    print(f"  Output: {OUTPUT_IMAGES_DIR}")

    # Step 2: Generate synthetic barcodes with realistic sizes
    class_mapping = {
        'box': 0,
        'empty_shelves': 1,
        'palletes': 2,
        'product_barcodes': 3,
        'shelf': 4,
        'shelf_barcodes': 5
    }

    # Size ranges from real data analysis
    shelf_barcode_size_range = (
        (0.0232, 0.1481),  # width: min-max
        (0.0235, 0.0904)   # height: min-max
    )

    product_barcode_size_range = (
        (0.0292, 0.1204),  # width: min-max
        (0.0445, 0.1520)   # height: min-max
    )

    generator = ImprovedWarehouseBarcodeGenerator(
        class_mapping=class_mapping,
        shelf_barcode_size_range=shelf_barcode_size_range,
        product_barcode_size_range=product_barcode_size_range
    )

    generator.process_dataset(
        images_dir=INPUT_IMAGES_DIR,
        labels_dir=TEMP_LABELS_DIR,
        output_images_dir=OUTPUT_IMAGES_DIR,
        output_labels_dir=OUTPUT_LABELS_DIR,
        num_augmentations_per_image=3  # 3 versions per image
    )

    print("\n" + "="*70)
    print("✅ COMPLETE!")
    print("="*70)
    print(f"\nYour improved synthetic dataset is ready:")
    print(f"  - Images: {OUTPUT_IMAGES_DIR}")
    print(f"  - Labels: {OUTPUT_LABELS_DIR}")
    print(f"\nDataset includes:")
    print(f"  - Code128 barcodes on shelves (class 5: shelf_barcodes)")
    print(f"  - DataMatrix barcodes on pallets (class 3: product_barcodes)")
    print(f"  - All with realistic sizes matching your training data!")


if __name__ == "__main__":
    main()
