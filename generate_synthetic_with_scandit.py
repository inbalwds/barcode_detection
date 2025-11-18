#!/usr/bin/env python3
"""
Complete pipeline for generating synthetic barcode dataset with Scandit integration

Usage:
    python3 generate_synthetic_with_scandit.py [--config CONFIG_PATH]

Example:
    python3 generate_synthetic_with_scandit.py --config config.yaml
"""

import os
import sys
import yaml
import argparse
from pathlib import Path

# Add project root to path
sys.path.append('/home/ubuntu/barcode_detection')
sys.path.append('/home/ubuntu/barcode_detection/data')

from data.improved_warehouse_generator import ImprovedWarehouseBarcodeGenerator


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Generate synthetic barcode dataset with Scandit L1 replacement'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='/home/ubuntu/barcode_detection/config.yaml',
        help='Path to configuration YAML file (default: config.yaml)'
    )
    parser.add_argument(
        '--input-images',
        type=str,
        help='Input images directory (overrides config)'
    )
    parser.add_argument(
        '--input-labels',
        type=str,
        help='Input labels directory (overrides config)'
    )
    parser.add_argument(
        '--output-images',
        type=str,
        help='Output images directory (overrides config)'
    )
    parser.add_argument(
        '--output-labels',
        type=str,
        help='Output labels directory (overrides config)'
    )
    parser.add_argument(
        '--num-augmentations',
        type=int,
        default=3,
        help='Number of augmented versions per image (default: 3)'
    )
    parser.add_argument(
        '--no-scandit',
        action='store_true',
        help='Disable Scandit detection (no L1 replacement)'
    )
    parser.add_argument(
        '--no-replace-l1',
        action='store_true',
        help='Disable L1 barcode replacement (keep Scandit detection for stats)'
    )

    return parser.parse_args()


def load_config(config_path):
    """Load configuration from YAML file"""
    if not os.path.exists(config_path):
        print(f"ERROR: Config file not found: {config_path}")
        sys.exit(1)

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    return config


def validate_paths(images_dir, labels_dir):
    """Validate input paths exist"""
    if not os.path.exists(images_dir):
        print(f"ERROR: Input images directory not found: {images_dir}")
        return False

    if not os.path.exists(labels_dir):
        print(f"WARNING: Input labels directory not found: {labels_dir}")
        print(f"  Creating empty labels directory...")
        os.makedirs(labels_dir, exist_ok=True)

    return True


def main():
    """Main execution function"""
    args = parse_args()

    print("="*80)
    print("SCANDIT-INTEGRATED SYNTHETIC BARCODE DATASET GENERATOR")
    print("="*80)
    print()

    # Load configuration
    print(f"Loading configuration from: {args.config}")
    config = load_config(args.config)
    print("✓ Configuration loaded")
    print()

    # Determine paths (command line args override config)
    input_images = args.input_images or config.get('INPUT_IMAGES_DIR')
    input_labels = args.input_labels or config.get('TEST_LABELS_DIR')
    output_images = args.output_images or '/home/ubuntu/barcode_detection/synthetic_output/images'
    output_labels = args.output_labels or '/home/ubuntu/barcode_detection/synthetic_output/labels'

    # Validate paths
    if not validate_paths(input_images, input_labels):
        return 1

    # Display configuration
    print("Configuration:")
    print(f"  Input Images:  {input_images}")
    print(f"  Input Labels:  {input_labels}")
    print(f"  Output Images: {output_images}")
    print(f"  Output Labels: {output_labels}")
    print(f"  Augmentations: {args.num_augmentations} per image")
    print(f"  Scandit:       {'Disabled' if args.no_scandit else 'Enabled'}")
    print(f"  L1 Replace:    {'Disabled' if args.no_replace_l1 else 'Enabled'}")
    print()

    # Class mapping
    class_mapping = {
        'box': 0,
        'empty_shelves': 1,
        'palletes': 2,
        'product_barcodes': 3,
        'shelf': 4,
        'shelf_barcodes': 5
    }

    # Get Scandit license key
    scandit_license = config.get('license_key')
    if not scandit_license and not args.no_scandit:
        print("WARNING: No Scandit license key found in config")
        print("  Proceeding without Scandit detection...")
        use_scandit = False
    else:
        use_scandit = not args.no_scandit

    # Initialize generator
    print("Initializing generator...")
    try:
        generator = ImprovedWarehouseBarcodeGenerator(
            class_mapping=class_mapping,
            scandit_license_key=scandit_license if use_scandit else None,
            use_scandit=use_scandit
        )
        print("✓ Generator initialized")
        print()
    except Exception as e:
        print(f"ERROR: Failed to initialize generator: {e}")
        return 1

    # Process dataset
    print("Starting dataset processing...")
    print()
    print("What will happen:")
    if use_scandit and not args.no_replace_l1:
        print("  1. Scandit will detect DataMatrix barcodes starting with 'L1'")
        print("  2. Each detected L1 barcode will be replaced with a new generated one")
        print("  3. Additional synthetic barcodes will be added:")
        print("     - Code128 barcodes on shelves")
        print("     - DataMatrix barcodes on pallets")
    else:
        print("  1. Synthetic barcodes will be added:")
        print("     - Code128 barcodes on shelves")
        print("     - DataMatrix barcodes on pallets")
    print()

    try:
        generator.process_dataset(
            images_dir=input_images,
            labels_dir=input_labels,
            output_images_dir=output_images,
            output_labels_dir=output_labels,
            num_augmentations_per_image=args.num_augmentations,
            replace_detected_l1=(not args.no_replace_l1)
        )
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        return 1
    except Exception as e:
        print(f"\nERROR during processing: {e}")
        import traceback
        traceback.print_exc()
        return 1

    print()
    print("="*80)
    print("✅ GENERATION COMPLETE!")
    print("="*80)
    print()
    print("Next steps:")
    print("  1. Review generated images and labels")
    print("  2. Verify barcode quality and placement")
    print("  3. Use synthetic dataset to train your YOLO model")
    print()
    print(f"Results saved to:")
    print(f"  - Images: {output_images}")
    print(f"  - Labels: {output_labels}")
    print()

    return 0


if __name__ == "__main__":
    exit(main())
