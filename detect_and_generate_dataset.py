"""
Detect pallets and shelves using trained YOLO model, then generate synthetic barcode dataset
"""

import os
import sys
from pathlib import Path
import cv2
import numpy as np
from typing import List, Tuple

# Add ultralytics
try:
    from ultralytics import YOLO
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'ultralytics', '--break-system-packages'])
    from ultralytics import YOLO

# Import the existing generator
from data.warehouse_data_generator import WarehouseBarcodeGenerator, YOLOAnnotation


class PalletShelfDetector:
    """
    Uses trained YOLO model to detect pallets and shelves in warehouse images
    """

    def __init__(self, model_path: str, class_names: List[str]):
        """
        Args:
            model_path: Path to trained YOLO weights (.pt file)
            class_names: List of class names matching the model
                        ['box', 'empty_shelves', 'palletes', 'product_barcodes', 'shelf', 'shelf_barcodes']
        """
        print(f"Loading model from {model_path}...")
        self.model = YOLO(model_path)
        self.class_names = class_names

        # Class IDs we care about for detection
        self.pallet_class = class_names.index('palletes') if 'palletes' in class_names else None
        self.shelf_class = class_names.index('shelf') if 'shelf' in class_names else None
        self.empty_shelves_class = class_names.index('empty_shelves') if 'empty_shelves' in class_names else None
        self.box_class = class_names.index('box') if 'box' in class_names else None

        print(f"✓ Model loaded successfully")
        print(f"  - Pallet class ID: {self.pallet_class}")
        print(f"  - Shelf class ID: {self.shelf_class}")
        print(f"  - Empty shelves class ID: {self.empty_shelves_class}")
        print(f"  - Box class ID: {self.box_class}")

    def detect_image(self,
                     image_path: str,
                     conf_threshold: float = 0.25,
                     iou_threshold: float = 0.45) -> List[YOLOAnnotation]:
        """
        Detect pallets, shelves, boxes in an image

        Args:
            image_path: Path to image
            conf_threshold: Confidence threshold for detection
            iou_threshold: IoU threshold for NMS

        Returns:
            List of YOLOAnnotation objects
        """
        # Run inference
        results = self.model.predict(
            image_path,
            conf=conf_threshold,
            iou=iou_threshold,
            verbose=False
        )

        annotations = []

        if len(results) > 0:
            result = results[0]

            # Get image dimensions
            img_h, img_w = result.orig_shape

            # Process each detection
            if result.boxes is not None:
                for box in result.boxes:
                    # Get box data
                    cls_id = int(box.cls[0].item())
                    conf = float(box.conf[0].item())

                    # Get xyxy format and convert to xywh
                    xyxy = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = xyxy

                    # Convert to bbox format (x, y, w, h)
                    x = int(x1)
                    y = int(y1)
                    w = int(x2 - x1)
                    h = int(y2 - y1)

                    # Convert to YOLO format
                    yolo_ann = YOLOAnnotation.from_bbox(
                        class_id=cls_id,
                        bbox=(x, y, w, h),
                        img_width=img_w,
                        img_height=img_h
                    )

                    annotations.append(yolo_ann)

        return annotations

    def save_annotations(self, annotations: List[YOLOAnnotation], output_path: str):
        """Save YOLO annotations to file"""
        with open(output_path, 'w') as f:
            for ann in annotations:
                f.write(ann.to_yolo_string() + '\n')

    def process_directory(self,
                         input_images_dir: str,
                         output_labels_dir: str,
                         conf_threshold: float = 0.25,
                         visualize: bool = False,
                         output_vis_dir: str = None):
        """
        Process all images in directory and save detections

        Args:
            input_images_dir: Directory containing warehouse images
            output_labels_dir: Directory to save YOLO format labels
            conf_threshold: Confidence threshold
            visualize: Whether to save visualization images
            output_vis_dir: Directory for visualizations (if visualize=True)
        """
        os.makedirs(output_labels_dir, exist_ok=True)

        if visualize:
            if output_vis_dir is None:
                output_vis_dir = Path(output_labels_dir).parent / 'visualizations'
            os.makedirs(output_vis_dir, exist_ok=True)

        # Get all images
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
            image_files.extend(Path(input_images_dir).glob(ext))

        print(f"\nDetecting pallets/shelves in {len(image_files)} images...")
        print(f"Confidence threshold: {conf_threshold}")

        stats = {
            'total': len(image_files),
            'processed': 0,
            'with_detections': 0,
            'total_detections': 0
        }

        for img_path in image_files:
            img_name = img_path.stem

            try:
                # Detect
                annotations = self.detect_image(str(img_path), conf_threshold=conf_threshold)

                # Save labels
                output_label_path = Path(output_labels_dir) / f"{img_name}.txt"
                self.save_annotations(annotations, str(output_label_path))

                # Update stats
                stats['processed'] += 1
                if len(annotations) > 0:
                    stats['with_detections'] += 1
                    stats['total_detections'] += len(annotations)

                # Count by class
                class_counts = {}
                for ann in annotations:
                    cls_name = self.class_names[ann.class_id]
                    class_counts[cls_name] = class_counts.get(cls_name, 0) + 1

                detection_str = ', '.join([f"{count} {cls}" for cls, count in class_counts.items()])
                print(f"✓ {img_name}: {len(annotations)} detections ({detection_str if detection_str else 'none'})")

                # Visualize (optional)
                if visualize and len(annotations) > 0:
                    img = cv2.imread(str(img_path))
                    img_h, img_w = img.shape[:2]

                    # Draw boxes
                    for ann in annotations:
                        x, y, w, h = ann.to_bbox(img_w, img_h)
                        cls_name = self.class_names[ann.class_id]

                        # Color by class
                        colors = {
                            'palletes': (255, 0, 0),      # Blue
                            'shelf': (0, 255, 0),          # Green
                            'empty_shelves': (0, 255, 255), # Yellow
                            'box': (255, 255, 0)           # Cyan
                        }
                        color = colors.get(cls_name, (255, 255, 255))

                        cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
                        cv2.putText(img, cls_name, (x, y-5),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                    # Save visualization
                    vis_path = Path(output_vis_dir) / f"{img_name}_detections.jpg"
                    cv2.imwrite(str(vis_path), img)

            except Exception as e:
                print(f"✗ Error processing {img_name}: {e}")

        print(f"\n{'='*60}")
        print(f"Detection Results:")
        print(f"  - Processed: {stats['processed']}/{stats['total']} images")
        print(f"  - With detections: {stats['with_detections']} images")
        print(f"  - Total detections: {stats['total_detections']}")
        print(f"  - Average per image: {stats['total_detections']/max(stats['processed'],1):.1f}")
        print(f"{'='*60}\n")

        return stats


def main():
    """
    Complete pipeline:
    1. Detect pallets/shelves in warehouse images
    2. Generate synthetic barcode dataset
    """

    # Configuration
    MODEL_PATH = '/home/ubuntu/barcode_detection/models/rami_levi_weights_with_first_floor.pt'
    INPUT_IMAGES_DIR = '/home/ubuntu/barcode_detection/rami levy.v16i.yolov12/train/images'

    # Temporary directory for detections
    TEMP_LABELS_DIR = '/home/ubuntu/barcode_detection/temp_detected_labels'

    # Output directories for synthetic dataset
    OUTPUT_IMAGES_DIR = '/home/ubuntu/barcode_detection/synthetic_dataset/images'
    OUTPUT_LABELS_DIR = '/home/ubuntu/barcode_detection/synthetic_dataset/labels'

    # Class names (must match your model)
    CLASS_NAMES = ['box', 'empty_shelves', 'palletes', 'product_barcodes', 'shelf', 'shelf_barcodes']

    print("="*60)
    print("WAREHOUSE SYNTHETIC BARCODE DATASET GENERATOR")
    print("="*60)
    print(f"\nStep 1: Detecting pallets and shelves...")
    print(f"  Model: {MODEL_PATH}")
    print(f"  Input: {INPUT_IMAGES_DIR}")
    print(f"  Temp labels: {TEMP_LABELS_DIR}")

    # Step 1: Detect pallets and shelves
    detector = PalletShelfDetector(
        model_path=MODEL_PATH,
        class_names=CLASS_NAMES
    )

    stats = detector.process_directory(
        input_images_dir=INPUT_IMAGES_DIR,
        output_labels_dir=TEMP_LABELS_DIR,
        conf_threshold=0.25,
        visualize=True  # Save visualizations to check detections
    )

    if stats['with_detections'] == 0:
        print("\n⚠ Warning: No detections found! Check your model and confidence threshold.")
        print("Tip: Try lowering the confidence threshold or verify your model is correct.")
        return

    print(f"\nStep 2: Generating synthetic barcode dataset...")
    print(f"  Output images: {OUTPUT_IMAGES_DIR}")
    print(f"  Output labels: {OUTPUT_LABELS_DIR}")

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
        datamatrix_enabled=False  # Only Code128 for now
    )

    generator.process_dataset(
        images_dir=INPUT_IMAGES_DIR,
        labels_dir=TEMP_LABELS_DIR,
        output_images_dir=OUTPUT_IMAGES_DIR,
        output_labels_dir=OUTPUT_LABELS_DIR,
        num_augmentations_per_image=3  # Create 3 versions of each image
    )

    print("\n" + "="*60)
    print("✅ COMPLETE!")
    print("="*60)
    print(f"\nYour synthetic dataset is ready:")
    print(f"  - Images: {OUTPUT_IMAGES_DIR}")
    print(f"  - Labels: {OUTPUT_LABELS_DIR}")
    print(f"\nYou can now use this dataset to train your barcode detection model!")


if __name__ == "__main__":
    main()
