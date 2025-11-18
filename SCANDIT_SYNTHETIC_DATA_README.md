# Scandit-Integrated Synthetic Barcode Generator

## Overview

This enhanced synthetic data generator integrates with the **Scandit Barcode Detection SDK** to create realistic training data for barcode detection models. The key feature is the ability to detect existing L1-prefixed DataMatrix barcodes in warehouse images and replace them with newly generated ones.

## Key Features

### 1. L1 DataMatrix Detection and Replacement
- Uses Scandit SDK to detect DataMatrix barcodes starting with `L1` pattern
- Replaces detected barcodes with newly generated L1-prefixed DataMatrix codes
- Preserves barcode position, size, and orientation using perspective transforms
- Applies realistic augmentations (blur, noise, lighting variations)

### 2. Additional Synthetic Barcode Placement
- **Code128 on shelves**: Realistic shelf_barcode sizes
- **DataMatrix on pallets**: Realistic product_barcode sizes
- Smart placement within ROI (Region of Interest)
- Overlap detection to prevent barcode collisions

### 3. L1 Barcode Generation
All generated DataMatrix barcodes follow the pattern:
```
L1{ALPHANUMERIC}
```
Example: `L1ABC123XYZ`, `L17K9M2P4Q`, `L1WAREHOUSE01`

## Architecture

### Main Components

1. **ImprovedWarehouseBarcodeGenerator** ([data/improved_warehouse_generator.py](data/improved_warehouse_generator.py))
   - Main generator class with Scandit integration
   - Methods:
     - `detect_and_extract_l1_barcodes()`: Detect L1 DataMatrix codes
     - `replace_barcode_in_image()`: Replace detected barcode with new one
     - `generate_synthetic_image()`: Complete pipeline
     - `process_dataset()`: Batch processing

2. **ScanditDecoder** ([scandit/barcode_detector.py](scandit/barcode_detector.py:30))
   - Wrapper around Scandit SDK
   - Multi-variant detection (rotation, CLAHE, blur, threshold)
   - Returns barcode data, symbology, and corner coordinates

## Usage

### Quick Test (Single Image)

```bash
python3 quick_test_scandit.py
```

This will:
1. Load the first image from your dataset
2. Detect any L1 DataMatrix codes using Scandit
3. Replace detected L1 codes with new generated ones
4. Add additional synthetic barcodes
5. Save result to `quick_test_output/`

### Full Dataset Processing

```bash
python3 test_scandit_synthetic_generator.py
```

Or programmatically:

```python
from data.improved_warehouse_generator import ImprovedWarehouseBarcodeGenerator
import yaml

# Load config
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

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
    scandit_license_key=config['license_key'],
    use_scandit=True  # Enable Scandit detection
)

# Process dataset
generator.process_dataset(
    images_dir='path/to/images',
    labels_dir='path/to/labels',
    output_images_dir='path/to/output/images',
    output_labels_dir='path/to/output/labels',
    num_augmentations_per_image=3,
    replace_detected_l1=True  # Enable L1 replacement
)
```

### Single Image Processing

```python
# Process a single image
syn_image, syn_annotations = generator.generate_synthetic_image(
    image_path='warehouse_image.jpg',
    annotation_path='warehouse_image.txt',
    num_barcodes_range=(2, 5),  # Add 2-5 additional barcodes
    replace_detected_l1=True     # Replace detected L1 codes
)

# Save results
import cv2
cv2.imwrite('output_image.jpg', syn_image)
generator.save_yolo_annotations(syn_annotations, 'output_label.txt')
```

## Configuration

### Required Files

1. **config.yaml** - Must contain:
```yaml
license_key: "YOUR_SCANDIT_LICENSE_KEY"
INPUT_IMAGES_DIR: '/path/to/images'
TEST_LABELS_DIR: '/path/to/labels'
CLASS_NAMES: ['box', 'empty_shelves', 'palletes', 'product_barcodes', 'shelf', 'shelf_barcodes']
```

2. **Input Data Structure**:
```
images/
  ├── image1.jpg
  ├── image2.jpg
  └── ...
labels/
  ├── image1.txt  # YOLO format annotations
  ├── image2.txt
  └── ...
```

### Barcode Size Ranges

The generator uses realistic size distributions from actual warehouse data:

- **Shelf Barcodes (Code128)**:
  - Width: 0.0232 - 0.1481 (normalized)
  - Height: 0.0235 - 0.0904 (normalized)

- **Product Barcodes (DataMatrix)**:
  - Width: 0.0292 - 0.1204 (normalized)
  - Height: 0.0445 - 0.1520 (normalized)

## How L1 Replacement Works

### Step-by-Step Process

1. **Detection Phase**:
   ```python
   l1_barcodes = self.detect_and_extract_l1_barcodes(image)
   # Returns: [{'data': 'L1ABC123', 'corners': [(x1,y1), ...], 'symbology': 'DATA_MATRIX'}]
   ```

2. **For Each Detected L1 Barcode**:
   - Generate new L1 text: `new_text = self.generate_random_text('datamatrix')`
   - Create DataMatrix image: `new_barcode_img = self.create_datamatrix(new_text, (100, 100))`
   - Apply augmentations: `new_barcode_img = self.apply_realistic_augmentations(new_barcode_img)`

3. **Replacement Phase**:
   ```python
   # Calculate bounding box from corners
   # Resize new barcode to match detected barcode size
   # Apply perspective transform to match orientation
   # Blend into original image
   ```

4. **Perspective Transform**:
   - Source corners: New barcode (rectangular)
   - Destination corners: Detected barcode corners (potentially skewed)
   - Result: New barcode matches position, size, and rotation of original

## Regex Pattern Matching

The L1 detection uses flexible regex matching:

```python
if re.match(r'^L1', data, re.IGNORECASE):
    # Matches: L1, l1, L1ABC, L1123, etc.
```

Symbology check:
```python
if "DATA_MATRIX" in symbology.upper() or "DATAMATRIX" in symbology.upper():
    # Matches various DataMatrix symbology strings
```

## Realistic Augmentations

Applied to all generated barcodes:

1. **Rotation**: ±15° (60% probability)
2. **Perspective**: Small perspective distortion (40% probability)
3. **Blur**: Gaussian or motion blur (50% probability)
4. **Lighting**: Contrast/brightness variations (60% probability)
5. **Noise**: Gaussian noise (40% probability)
6. **Occlusion**: Partial occlusion (15% probability)

## Output

### Generated Files

For each input image with `num_augmentations_per_image=3`:
- `image_syn_0.jpg`, `image_syn_0.txt`
- `image_syn_1.jpg`, `image_syn_1.txt`
- `image_syn_2.jpg`, `image_syn_2.txt`

### YOLO Format Labels

```
class_id x_center y_center width height
5 0.4532 0.6234 0.0823 0.0512
3 0.7821 0.3421 0.0612 0.0923
```

Where:
- `class_id 5` = shelf_barcodes (Code128)
- `class_id 3` = product_barcodes (DataMatrix)

## Performance

- **Detection**: ~50-200ms per image (depends on image size and barcode count)
- **Generation**: ~100-300ms per image
- **Total**: ~150-500ms per synthetic image

## Troubleshooting

### Import Errors

If you get `ModuleNotFoundError`:
```bash
# Make sure scandit SDK is available
ls /home/ubuntu/barcode_detection/scandit/scanditsdk.py

# Test import
python3 -c "from scandit.barcode_detector import ScanditDecoder"
```

### No Barcodes Detected

- Check Scandit license key in config.yaml
- Verify images contain DataMatrix barcodes starting with L1
- Try lower confidence threshold
- Check image quality (resolution, focus, lighting)

### Barcode Replacement Fails

- Check corner coordinates are valid (4 corners)
- Verify image is large enough for perspective transform
- Check barcode size is reasonable (not too small)

## Example Output

**Console Output**:
```
Processing 50 images...
  - Scandit detection enabled: will detect and replace L1 DataMatrix codes

  Found 3 L1 DataMatrix codes to replace
    ✓ Replaced 'L1ABC123' with 'L17X9K2M4'
    ✓ Replaced 'L1XYZ789' with 'L1P5Q8W3Y'
    ✓ Replaced 'L1PROD01' with 'L1M9N2K7L'
  Successfully replaced 3/3 L1 barcodes

✓ image1_syn_0 (+2 Code128, +3 DataMatrix)
...
```

## Next Steps

1. **Review Generated Images**: Check `output_images_dir` for quality
2. **Verify Labels**: Ensure YOLO annotations are correct
3. **Train YOLO Model**: Use synthetic dataset for training
4. **Fine-tune**: Adjust size ranges, augmentation probabilities as needed

## Related Files

- [data/improved_warehouse_generator.py](data/improved_warehouse_generator.py) - Main generator
- [scandit/barcode_detector.py](scandit/barcode_detector.py) - Scandit wrapper
- [quick_test_scandit.py](quick_test_scandit.py) - Quick test script
- [test_scandit_synthetic_generator.py](test_scandit_synthetic_generator.py) - Full test script
- [config.yaml](config.yaml) - Configuration file

## References

- Scandit Barcode Scanner SDK Documentation
- YOLO Object Detection Format
- OpenCV Perspective Transforms
- DataMatrix Barcode Specification
