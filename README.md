# Rami Levi Warehouse Barcode Detection

Synthetic dataset generator for warehouse barcode detection using Scandit SDK and YOLO format annotations.

## Overview

This project generates synthetic training data for barcode detection in warehouse environments. It uses Scandit SDK to detect existing barcodes in warehouse images and replaces them with newly generated barcodes at exact locations, maintaining original orientation and size.

### Features

- **Scandit Integration**: Detects DataMatrix and Code128 barcodes in warehouse images
- **Exact Placement**: Replaces barcodes at detected locations using perspective transforms
- **Pattern-Based Generation**: Follows Rami Levi barcode patterns
  - Product barcodes (DataMatrix): `L1000000000` + 9 random digits
  - Shelf barcodes (Code128): 4 digits + `1` + 2 digits
- **Hybrid Mode**: Detects and replaces existing barcodes, optionally adds random ones
- **YOLO Format**: Generates annotations compatible with YOLO object detection
- **Progress Tracking**: Real-time statistics every 10 frames

## Quick Start

### Prerequisites

```bash
# System dependencies
sudo apt-get install libdmtx0b libdmtx-dev

# Python dependencies
pip install -r requirements.txt
```

### Basic Usage

```bash
# Generate synthetic dataset
python3 generate_synthetic_with_scandit.py \
    --input-images ./input_images \
    --input-labels ./input_labels \
    --output-images ./output/images \
    --output-labels ./output/labels \
    --num-augmentations 5
```

### Quick Test

```bash
# Test on a single image
python3 quick_test_scandit.py
```

## Barcode Patterns

### Product Barcode (DataMatrix)
- **Pattern**: `^L1(0{9}\d{9})$`
- **Format**: `L1` + `000000000` + 9 random digits
- **Example**: `L1000000000123456789`
- **Length**: 20 characters

### Shelf Barcode (Code128)
- **Pattern**: `^\d{4}1\d{2}$`
- **Format**: 4 digits + `1` + 2 digits
- **Example**: `2203156`
- **Length**: 7 digits

See RAMI_LEVI_BARCODE_PATTERNS.md for detailed pattern documentation.

## Project Structure

```
barcode_detection/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── config.yaml                        # Scandit license key
├── .gitignore                         # Git ignore rules
│
├── data/
│   └── improved_warehouse_generator.py  # Main generator class
│
├── scandit/
│   └── barcode_detector.py            # Scandit SDK wrapper
│
├── generate_synthetic_with_scandit.py # Main script
├── quick_test_scandit.py              # Quick test script
└── test_rami_levi_patterns.py         # Pattern validation
```

## Configuration

Edit `config.yaml` to set your Scandit license key.

## Testing

### Test Pattern Generation

```bash
python3 test_rami_levi_patterns.py
```

### Quick Test on Sample Image

```bash
python3 quick_test_scandit.py
```

## Progress Reporting

The generator prints progress reports every 10 frames showing:
- DataMatrix detected
- Barcodes added
- Images processed

## Dependencies

- Python 3.8+
- PyTorch 2.0+
- OpenCV 4.8+
- Scandit SDK (included)
- libdmtx (system library)

See requirements.txt for complete list.
