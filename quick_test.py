"""
Quick test - Process just a few images to verify the pipeline works
"""

import os
import sys
from pathlib import Path

# Import the existing components
from detect_and_generate_dataset import PalletShelfDetector
from data.warehouse_data_generator import WarehouseBarcodeGenerator


def main():
    """Test pipeline on a small subset of images"""

    # Configuration
    MODEL_PATH = '/home/ubuntu/barcode_detection/models/rami_levi_weights_with_first_floor.pt'
    INPUT_IMAGES_DIR = '/home/ubuntu/barcode_detection/rami levy.v16i.yolov12/train/images'

    # Test output directories
    TEST_LABELS_DIR = '/home/ubuntu/barcode_detection/test_output/detected_labels'
    TEST_VIS_DIR = '/home/ubuntu/barcode_detection/test_output/visualizations'
    TEST_SYNTHETIC_IMAGES = '/home/ubuntu/barcode_detection/test_output/synthetic_images'
    TEST_SYNTHETIC_LABELS = '/home/ubuntu/barcode_detection/test_output/synthetic_labels'

    # Class names
    CLASS_NAMES = ['box', 'empty_shelves', 'palletes', 'product_barcodes', 'shelf', 'shelf_barcodes']

    print("="*60)
    print("QUICK TEST - Processing 5 images")
    print("="*60)

    # Get first 5 images
    image_files = list(Path(INPUT_IMAGES_DIR).glob('*.jpg'))[:5]

    if not image_files:
        print("No images found!")
        return

    # Create a temp directory for just these images
    temp_images_dir = '/home/ubuntu/barcode_detection/test_output/temp_images'
    os.makedirs(temp_images_dir, exist_ok=True)

    # Copy first 5 images
    import shutil
    for img_path in image_files:
        shutil.copy(str(img_path), temp_images_dir)

    print(f"\nStep 1: Detecting pallets/shelves in {len(image_files)} images...")

    # Step 1: Detect
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

    if stats['with_detections'] == 0:
        print("\nNo detections found!")
        return

    print(f"\nStep 2: Generating synthetic barcodes...")

    # Step 2: Generate synthetic barcodes
    class_mapping = {
        'box': 0,
        'empty_shelves': 1,
        'palletes': 2,
        'product_barcodes': 3,
        'shelf': 4,
        'shelf_barcodes': 5
    }

    generator = WarehouseBarcodeGenerator(
        class_mapping=class_mapping,
        datamatrix_enabled=False
    )

    generator.process_dataset(
        images_dir=temp_images_dir,
        labels_dir=TEST_LABELS_DIR,
        output_images_dir=TEST_SYNTHETIC_IMAGES,
        output_labels_dir=TEST_SYNTHETIC_LABELS,
        num_augmentations_per_image=2  # Create 2 versions of each
    )

    print("\n" + "="*60)
    print("TEST COMPLETE!")
    print("="*60)
    print(f"\nResults:")
    print(f"  - Detection visualizations: {TEST_VIS_DIR}")
    print(f"  - Synthetic images: {TEST_SYNTHETIC_IMAGES}")
    print(f"  - Synthetic labels: {TEST_SYNTHETIC_LABELS}")
    print("\nCheck the visualizations to see detected pallets/shelves")
    print("and synthetic images to see the added barcodes!")


if __name__ == "__main__":
    main()
