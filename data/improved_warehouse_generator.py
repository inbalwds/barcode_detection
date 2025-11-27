"""
Improved Warehouse Synthetic Barcode Data Generator
- Code128 on shelves (sized like real shelf_barcodes)
- DataMatrix on pallets (sized like real product_barcodes)
- Integrates with Scandit detector to replace detected L1-prefixed DataMatrix codes
"""

import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
import random
import string
import os
import re
import sys
from pathlib import Path
from typing import Tuple, Dict, List, Optional
from dataclasses import dataclass
import json

try:
    import barcode
    from barcode.writer import ImageWriter
except ImportError:
    import subprocess
    subprocess.check_call(['pip', 'install', 'python-barcode', '--break-system-packages'])
    import barcode
    from barcode.writer import ImageWriter

# Import base classes from original generator
sys.path.append('/home/ubuntu/barcode_detection')
from data.warehouse_data_generator import YOLOAnnotation

# Import Scandit detector
sys.path.append('/home/ubuntu/barcode_detection/scandit')
try:
    from barcode_detector import ScanditDecoder
    SCANDIT_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import ScanditDecoder: {e}")
    SCANDIT_AVAILABLE = False


class ImprovedWarehouseBarcodeGenerator:
    """
    Improved generator that:
    - Places Code128 on shelves with realistic shelf_barcode sizes
    - Places DataMatrix on pallets with realistic product_barcode sizes
    """

    def __init__(self,
                 class_mapping: Dict[str, int],
                 shelf_barcode_size_range: Tuple[Tuple[float, float], Tuple[float, float]] = None,
                 product_barcode_size_range: Tuple[Tuple[float, float], Tuple[float, float]] = None,
                 scandit_license_key: str = None,
                 use_scandit: bool = True):
        """
        Args:
            class_mapping: Mapping of class names to IDs
            shelf_barcode_size_range: ((min_w, max_w), (min_h, max_h)) in normalized coordinates
            product_barcode_size_range: ((min_w, max_w), (min_h, max_h)) in normalized coordinates
            scandit_license_key: Scandit license key for barcode detection
            use_scandit: Whether to use Scandit detection to replace barcodes
        """
        self.class_mapping = class_mapping

        # Initialize Scandit if requested and available
        self.scandit_decoder = None
        self.use_scandit = use_scandit and SCANDIT_AVAILABLE
        if self.use_scandit and scandit_license_key:
            try:
                self.scandit_decoder = ScanditDecoder(scandit_license_key)
                print("✓ Scandit detector initialized successfully")
            except Exception as e:
                print(f"Warning: Could not initialize Scandit decoder: {e}")
                self.use_scandit = False

        # Class IDs
        self.box_class = class_mapping.get('box', 0)
        self.empty_shelves_class = class_mapping.get('empty_shelves', 1)
        self.pallet_class = class_mapping.get('palletes', 2)
        self.product_barcode_class = class_mapping.get('product_barcodes', 3)
        self.shelf_class = class_mapping.get('shelf', 4)
        self.shelf_barcode_class = class_mapping.get('shelf_barcodes', 5)

        # Barcode size ranges (from real data analysis)
        # Shelf barcodes: avg 0.0726 × 0.0461, range 0.0232-0.1481 × 0.0235-0.0904
        self.shelf_barcode_size_range = shelf_barcode_size_range or (
            (0.0232, 0.1481),  # width range
            (0.0235, 0.0904)   # height range
        )

        # Product barcodes: avg 0.0626 × 0.0808, range 0.0292-0.1204 × 0.0445-0.1520
        self.product_barcode_size_range = product_barcode_size_range or (
            (0.0292, 0.1204),  # width range
            (0.0445, 0.1520)   # height range
        )

    def load_yolo_annotations(self, txt_path: str) -> List[YOLOAnnotation]:
        """Load YOLO annotations from txt file"""
        annotations = []

        if not os.path.exists(txt_path):
            return annotations

        with open(txt_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])

                    annotations.append(YOLOAnnotation(
                        class_id, x_center, y_center, width, height
                    ))

        return annotations

    def save_yolo_annotations(self, annotations: List[YOLOAnnotation], txt_path: str):
        """Save YOLO annotations to txt file"""
        with open(txt_path, 'w') as f:
            for ann in annotations:
                f.write(ann.to_yolo_string() + '\n')

    def generate_random_text(self, barcode_type: str) -> str:
        """Generate random text for barcode following Rami Levi patterns"""
        if barcode_type == 'code128':
            # Shelf barcode pattern: ^\d{4}1\d{2}$
            # Format: 4 digits + '1' + 2 digits = 7 digits total
            first_four = ''.join([str(random.randint(0, 9)) for _ in range(4)])
            last_two = ''.join([str(random.randint(0, 9)) for _ in range(2)])
            return f"{first_four}1{last_two}"
        else:  # datamatrix
            # Product barcode pattern: ^L1(0{9}\d{9})$
            # Format: L1 + 9 zeros + 9 random digits = 20 chars total
            nine_digits = ''.join([str(random.randint(0, 9)) for _ in range(9)])
            return f"L1000000000{nine_digits}"

    def create_code128(self, text: str, target_size: Tuple[int, int]) -> np.ndarray:
        """
        Create Code128 barcode image at target size

        Args:
            text: Barcode text
            target_size: (width, height) in pixels
        """
        code128 = barcode.get_barcode_class('code128')
        writer = ImageWriter()
        writer.dpi = 300

        options = {
            'module_width': 0.3,
            'module_height': 10.0,
            'quiet_zone': 2.0,
            'font_size': 0,
            'text_distance': 1.0,
            'background': 'white',
            'foreground': 'black',
        }

        barcode_obj = code128(text, writer=writer)

        import io
        buffer = io.BytesIO()
        barcode_obj.write(buffer, options=options)
        buffer.seek(0)

        pil_image = Image.open(buffer)
        img_array = np.array(pil_image.convert('RGB'))

        # Resize to target size
        img_resized = cv2.resize(img_array, target_size)

        return img_resized

    def create_datamatrix(self, text: str, target_size: Tuple[int, int]) -> np.ndarray:
        """
        Create DataMatrix barcode image at target size

        Since libdmtx is hard to install, we'll create a QR-code-like pattern
        that looks like DataMatrix for training purposes
        """
        # Create white background
        size = max(target_size)
        img = Image.new('RGB', (size, size), 'white')
        draw = ImageDraw.Draw(img)

        # Create DataMatrix-like pattern (grid of black/white squares)
        # DataMatrix has characteristic L-shaped finder pattern
        module_size = max(2, size // 20)  # Size of each square

        # Draw finder pattern (L-shape on two edges)
        # Top edge
        for i in range(0, size, module_size * 2):
            draw.rectangle([i, 0, i + module_size, module_size], fill='black')

        # Left edge
        for i in range(0, size, module_size * 2):
            draw.rectangle([0, i, module_size, i + module_size], fill='black')

        # Random data pattern inside
        for y in range(module_size * 2, size - module_size, module_size):
            for x in range(module_size * 2, size - module_size, module_size):
                if random.random() > 0.5:
                    draw.rectangle([x, y, x + module_size, y + module_size], fill='black')

        # Resize to target size
        img_resized = img.resize(target_size, Image.Resampling.NEAREST)
        img_array = np.array(img_resized)

        return img_array

    def apply_realistic_augmentations(self, barcode_img: np.ndarray) -> np.ndarray:
        """Apply realistic warehouse augmentations"""
        img = barcode_img.copy()

        # 1. Slight rotation
        if random.random() < 0.6:
            angle = random.uniform(-15, 15)
            h, w = img.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            img = cv2.warpAffine(img, M, (w, h),
                                borderMode=cv2.BORDER_CONSTANT,
                                borderValue=(255, 255, 255))

        # 2. Perspective
        if random.random() < 0.4:
            h, w = img.shape[:2]
            pts1 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
            offset = int(min(w, h) * 0.08)
            pts2 = np.float32([
                [random.randint(-offset, offset), random.randint(-offset, offset)],
                [w + random.randint(-offset, offset), random.randint(-offset, offset)],
                [random.randint(-offset, offset), h + random.randint(-offset, offset)],
                [w + random.randint(-offset, offset), h + random.randint(-offset, offset)]
            ])
            M = cv2.getPerspectiveTransform(pts1, pts2)
            img = cv2.warpPerspective(img, M, (w, h),
                                     borderMode=cv2.BORDER_CONSTANT,
                                     borderValue=(255, 255, 255))

        # 3. Blur
        if random.random() < 0.5:
            blur_type = random.choice(['gaussian', 'motion'])
            if blur_type == 'gaussian':
                kernel_size = random.choice([3, 5])
                img = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
            else:  # motion blur
                kernel_size = random.randint(3, 7)
                kernel = np.zeros((kernel_size, kernel_size))
                kernel[int((kernel_size-1)/2), :] = np.ones(kernel_size)
                kernel = kernel / kernel_size
                img = cv2.filter2D(img, -1, kernel)

        # 4. Lighting
        if random.random() < 0.6:
            alpha = random.uniform(0.6, 1.4)  # Contrast
            beta = random.randint(-40, 40)    # Brightness
            img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

        # 5. Noise
        if random.random() < 0.4:
            noise = np.random.normal(0, 15, img.shape).astype(np.uint8)
            img = cv2.add(img, noise)

        # 6. Partial occlusion
        if random.random() < 0.15:
            h, w = img.shape[:2]
            x1 = random.randint(0, w - w//5)
            y1 = random.randint(0, h - h//5)
            x2 = x1 + random.randint(w//15, w//5)
            y2 = y1 + random.randint(h//15, h//5)
            color = tuple(random.randint(50, 200) for _ in range(3))
            cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)

        return img

    def check_overlap(self, new_bbox: Tuple[float, float, float, float],
                     existing_annotations: List[YOLOAnnotation]) -> bool:
        """Check if new bbox overlaps significantly with existing ones"""
        new_x, new_y, new_w, new_h = new_bbox

        for ann in existing_annotations:
            # Calculate IoU
            x1 = max(new_x, ann.x_center - ann.width/2)
            y1 = max(new_y, ann.y_center - ann.height/2)
            x2 = min(new_x + new_w, ann.x_center + ann.width/2)
            y2 = min(new_y + new_h, ann.y_center + ann.height/2)

            if x2 > x1 and y2 > y1:
                intersection = (x2 - x1) * (y2 - y1)
                area1 = new_w * new_h
                area2 = ann.width * ann.height
                union = area1 + area2 - intersection
                iou = intersection / union if union > 0 else 0

                if iou > 0.3:  # Significant overlap
                    return True

        return False

    def detect_all_barcodes(self, image: np.ndarray) -> Dict[str, List[Dict]]:
        """
        Use Scandit to detect all barcodes and categorize them

        Returns:
            Dict with keys:
            - 'datamatrix_l1': DataMatrix codes starting with L1
            - 'code128': Code128 barcodes (shelf barcodes)
            - 'other': Other barcodes
        """
        if not self.use_scandit or self.scandit_decoder is None:
            return {'datamatrix_l1': [], 'code128': [], 'other': []}

        results = self.scandit_decoder.decode_image(image)

        categorized = {
            'datamatrix_l1': [],
            'code128': [],
            'other': []
        }

        for result in results:
            data = result.get("Decoded Data", "")
            symbology = result.get("Symbology", "")
            corners = result.get("Corners", [])

            barcode_info = {
                'data': data,
                'corners': corners,
                'symbology': symbology
            }

            # Categorize by type (check both original Scandit format and mapped format)
            symbology_normalized = symbology.upper().replace(" ", "").replace("-", "").replace("_", "")

            if "DATAMATRIX" in symbology_normalized:
                if re.match(r'^L1', data, re.IGNORECASE):
                    categorized['datamatrix_l1'].append(barcode_info)
                else:
                    categorized['other'].append(barcode_info)
            elif "CODE128" in symbology_normalized:
                categorized['code128'].append(barcode_info)
            else:
                categorized['other'].append(barcode_info)

        return categorized

    def replace_barcode_in_image(self,
                                 image: np.ndarray,
                                 barcode_corners: List[Tuple[int, int]],
                                 new_barcode_img: np.ndarray) -> Optional[np.ndarray]:
        """
        Replace a detected barcode with a new generated barcode

        Args:
            image: Original image
            barcode_corners: List of 4 corner points [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]
            new_barcode_img: New barcode image to place

        Returns:
            Modified image or None if replacement failed
        """
        img = image.copy()

        if len(barcode_corners) != 4:
            return None

        # Convert corners to numpy array
        corners = np.array(barcode_corners, dtype=np.float32)

        # Get bounding box of the detected barcode
        x_coords = corners[:, 0]
        y_coords = corners[:, 1]

        x_min, x_max = int(np.min(x_coords)), int(np.max(x_coords))
        y_min, y_max = int(np.min(y_coords)), int(np.max(y_coords))

        width = x_max - x_min
        height = y_max - y_min

        if width <= 0 or height <= 0:
            return None

        # Resize new barcode to match detected barcode size
        resized_barcode = cv2.resize(new_barcode_img, (width, height))

        # Apply perspective transform to match the orientation of detected barcode
        # Source corners: new barcode (rectangular, clockwise from top-left)
        h_bc, w_bc = resized_barcode.shape[:2]
        src_corners = np.array([
            [0, 0],
            [w_bc - 1, 0],
            [w_bc - 1, h_bc - 1],
            [0, h_bc - 1]
        ], dtype=np.float32)

        # Destination corners: detected barcode normalized to bounding box
        dst_corners_normalized = (corners - np.array([x_min, y_min])).astype(np.float32)

        # Get perspective transform (from src to dst)
        M = cv2.getPerspectiveTransform(src_corners, dst_corners_normalized)

        # Warp the new barcode
        warped_barcode = cv2.warpPerspective(
            resized_barcode, M, (width, height),
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(255, 255, 255)
        )

        # Blend the warped barcode into the image
        roi = img[y_min:y_max, x_min:x_max]
        if roi.shape[:2] == warped_barcode.shape[:2]:
            # Use alpha blending
            alpha = 0.9
            blended = cv2.addWeighted(warped_barcode, alpha, roi, 1 - alpha, 0)
            img[y_min:y_max, x_min:x_max] = blended
        else:
            return None

        return img

    def place_barcode_on_roi(self,
                            image: np.ndarray,
                            barcode_img: np.ndarray,
                            roi_ann: YOLOAnnotation,
                            img_w: int,
                            img_h: int,
                            blend: bool = True) -> Tuple[Optional[np.ndarray], Optional[Tuple[float, float, float, float]]]:
        """
        Place barcode within ROI (shelf or pallet)

        Returns:
            (modified_image, normalized_bbox) or (None, None) if placement failed
        """
        img = image.copy()

        # Get ROI bounds in pixels
        roi_x, roi_y, roi_w, roi_h = roi_ann.to_bbox(img_w, img_h)

        # Barcode size
        bc_h, bc_w = barcode_img.shape[:2]

        # Check if barcode fits in ROI
        if bc_w > roi_w * 0.9 or bc_h > roi_h * 0.9:
            # Barcode too big for ROI, skip
            return None, None

        # Random position within ROI
        margin = 0.1  # 10% margin from edges
        x_margin = int(roi_w * margin)
        y_margin = int(roi_h * margin)

        if roi_w - 2*x_margin < bc_w or roi_h - 2*y_margin < bc_h:
            return None, None

        x = random.randint(roi_x + x_margin, roi_x + roi_w - x_margin - bc_w)
        y = random.randint(roi_y + y_margin, roi_y + roi_h - y_margin - bc_h)

        # Ensure within image bounds
        x = max(0, min(x, img_w - bc_w))
        y = max(0, min(y, img_h - bc_h))

        # Blending
        if blend:
            alpha = random.uniform(0.85, 1.0)
            roi = img[y:y+bc_h, x:x+bc_w]
            if roi.shape[:2] == barcode_img.shape[:2]:
                blended = cv2.addWeighted(barcode_img, alpha, roi, 1-alpha, 0)
                img[y:y+bc_h, x:x+bc_w] = blended
            else:
                return None, None
        else:
            img[y:y+bc_h, x:x+bc_w] = barcode_img

        # Return normalized bbox
        norm_x = (x + bc_w/2) / img_w
        norm_y = (y + bc_h/2) / img_h
        norm_w = bc_w / img_w
        norm_h = bc_h / img_h

        return img, (norm_x, norm_y, norm_w, norm_h)

    def generate_synthetic_image(self,
                                image_path: str,
                                annotation_path: str,
                                num_barcodes_range: Tuple[int, int] = (2, 5),
                                replace_detected: bool = True,
                                add_random_barcodes: bool = False) -> Tuple[np.ndarray, List[YOLOAnnotation]]:
        """
        Generate synthetic image with realistically-sized barcodes

        Strategy:
        1. If Scandit enabled and replace_detected=True:
           - Detect existing barcodes (L1 DataMatrix + Code128)
           - Replace them with new generated barcodes AT THE SAME LOCATION
        2. If add_random_barcodes=True:
           - Place additional random barcodes on shelves/pallets
        """
        # Load image and annotations
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Cannot load image: {image_path}")

        img_h, img_w = image.shape[:2]
        existing_annotations = self.load_yolo_annotations(annotation_path)

        # Copy annotations (we'll add barcode annotations)
        new_annotations = existing_annotations.copy()

        # Step 1: Detect and replace existing barcodes at their detected locations
        replaced_dm_count = 0
        replaced_c128_count = 0

        if replace_detected and self.use_scandit:
            detected_barcodes = self.detect_all_barcodes(image)

            total_detected = len(detected_barcodes['datamatrix_l1']) + len(detected_barcodes['code128'])
            if total_detected > 0:
                print(f"  Found {len(detected_barcodes['datamatrix_l1'])} L1 DataMatrix + {len(detected_barcodes['code128'])} Code128 barcodes")

            # Replace L1 DataMatrix codes
            for barcode_info in detected_barcodes['datamatrix_l1']:
                new_text = self.generate_random_text('datamatrix')
                new_barcode_img = self.create_datamatrix(new_text, (100, 100))

                result = self.replace_barcode_in_image(
                    image,
                    barcode_info['corners'],
                    new_barcode_img
                )

                if result is not None:
                    image = result
                    replaced_dm_count += 1

                    # Add annotation for the replaced barcode
                    corners = np.array(barcode_info['corners'])
                    x_min, x_max = int(np.min(corners[:, 0])), int(np.max(corners[:, 0]))
                    y_min, y_max = int(np.min(corners[:, 1])), int(np.max(corners[:, 1]))

                    # Convert to YOLO format
                    x_center = ((x_min + x_max) / 2) / img_w
                    y_center = ((y_min + y_max) / 2) / img_h
                    width = (x_max - x_min) / img_w
                    height = (y_max - y_min) / img_h

                    yolo_ann = YOLOAnnotation(
                        self.product_barcode_class,
                        x_center, y_center, width, height
                    )
                    new_annotations.append(yolo_ann)

                    print(f"    ✓ Replaced L1 DataMatrix '{barcode_info['data']}' with '{new_text}'")
                else:
                    print(f"    ✗ Failed to replace '{barcode_info['data']}'")

            # Replace Code128 shelf barcodes
            for barcode_info in detected_barcodes['code128']:
                new_text = self.generate_random_text('code128')
                new_barcode_img = self.create_code128(new_text, (150, 50))

                result = self.replace_barcode_in_image(
                    image,
                    barcode_info['corners'],
                    new_barcode_img
                )

                if result is not None:
                    image = result
                    replaced_c128_count += 1

                    # Add annotation for the replaced barcode
                    corners = np.array(barcode_info['corners'])
                    x_min, x_max = int(np.min(corners[:, 0])), int(np.max(corners[:, 0]))
                    y_min, y_max = int(np.min(corners[:, 1])), int(np.max(corners[:, 1]))

                    # Convert to YOLO format
                    x_center = ((x_min + x_max) / 2) / img_w
                    y_center = ((y_min + y_max) / 2) / img_h
                    width = (x_max - x_min) / img_w
                    height = (y_max - y_min) / img_h

                    yolo_ann = YOLOAnnotation(
                        self.shelf_barcode_class,
                        x_center, y_center, width, height
                    )
                    new_annotations.append(yolo_ann)

                    print(f"    ✓ Replaced Code128 '{barcode_info['data']}' with '{new_text}'")
                else:
                    print(f"    ✗ Failed to replace '{barcode_info['data']}'")

            if total_detected > 0:
                print(f"  Successfully replaced {replaced_dm_count + replaced_c128_count}/{total_detected} barcodes")

        # Step 2: Optionally add random barcodes (only if requested)
        if add_random_barcodes:
            # Separate by type
            shelves = [ann for ann in existing_annotations if ann.class_id == self.shelf_class]
            pallets = [ann for ann in existing_annotations if ann.class_id == self.pallet_class]

            num_barcodes = random.randint(*num_barcodes_range)

            attempts = 0
            max_attempts = num_barcodes * 3
            added = 0

            print(f"  Adding {num_barcodes} random barcodes...")

            while added < num_barcodes and attempts < max_attempts:
                attempts += 1

                # Choose: Code128 on shelf or DataMatrix on pallet
                if len(shelves) > 0 and len(pallets) > 0:
                    barcode_type = random.choice(['code128_shelf', 'datamatrix_pallet'])
                elif len(shelves) > 0:
                    barcode_type = 'code128_shelf'
                elif len(pallets) > 0:
                    barcode_type = 'datamatrix_pallet'
                else:
                    break

                if barcode_type == 'code128_shelf':
                    # Code128 on shelf with shelf_barcode size
                    shelf = random.choice(shelves)

                    # Sample size from shelf_barcode distribution
                    norm_w = random.uniform(*self.shelf_barcode_size_range[0])
                    norm_h = random.uniform(*self.shelf_barcode_size_range[1])

                    target_w = int(norm_w * img_w)
                    target_h = int(norm_h * img_h)

                    if target_w < 10 or target_h < 10:
                        continue

                    text = self.generate_random_text('code128')
                    barcode_img = self.create_code128(text, (target_w, target_h))
                    barcode_img = self.apply_realistic_augmentations(barcode_img)

                    result, bbox = self.place_barcode_on_roi(
                        image, barcode_img, shelf, img_w, img_h
                    )

                    if result is not None:
                        # Check overlap
                        if not self.check_overlap(bbox, new_annotations):
                            image = result
                            yolo_ann = YOLOAnnotation(
                                self.shelf_barcode_class,
                                bbox[0], bbox[1], bbox[2], bbox[3]
                            )
                            new_annotations.append(yolo_ann)
                            added += 1

                elif barcode_type == 'datamatrix_pallet':
                    # DataMatrix on pallet with product_barcode size
                    pallet = random.choice(pallets)

                    # Sample size from product_barcode distribution
                    norm_w = random.uniform(*self.product_barcode_size_range[0])
                    norm_h = random.uniform(*self.product_barcode_size_range[1])

                    target_w = int(norm_w * img_w)
                    target_h = int(norm_h * img_h)

                    if target_w < 10 or target_h < 10:
                        continue

                    text = self.generate_random_text('datamatrix')
                    barcode_img = self.create_datamatrix(text, (target_w, target_h))
                    barcode_img = self.apply_realistic_augmentations(barcode_img)

                    result, bbox = self.place_barcode_on_roi(
                        image, barcode_img, pallet, img_w, img_h
                    )

                    if result is not None:
                        # Check overlap
                        if not self.check_overlap(bbox, new_annotations):
                            image = result
                            yolo_ann = YOLOAnnotation(
                                self.product_barcode_class,
                                bbox[0], bbox[1], bbox[2], bbox[3]
                            )
                            new_annotations.append(yolo_ann)
                            added += 1

        return image, new_annotations

    def process_dataset(self,
                       images_dir: str,
                       labels_dir: str,
                       output_images_dir: str,
                       output_labels_dir: str,
                       num_augmentations_per_image: int = 3,
                       replace_detected: bool = True,
                       add_random_barcodes: bool = False):
        """
        Process entire dataset

        Args:
            images_dir: Input images directory
            labels_dir: Input labels directory
            output_images_dir: Output images directory
            output_labels_dir: Output labels directory
            num_augmentations_per_image: Number of augmented versions per image
            replace_detected: Whether to detect and replace barcodes at their detected locations
            add_random_barcodes: Whether to add additional random barcodes
        """
        os.makedirs(output_images_dir, exist_ok=True)
        os.makedirs(output_labels_dir, exist_ok=True)

        image_files = list(Path(images_dir).glob('*.jpg')) + \
                     list(Path(images_dir).glob('*.png'))

        print(f"Processing {len(image_files)} images...")
        if self.use_scandit and replace_detected:
            print(f"  - Scandit detection enabled: will detect and replace barcodes at detected locations")
        if add_random_barcodes:
            print(f"  - Random barcode placement: enabled")

        stats = {
            'processed': 0,
            'code128_added': 0,
            'datamatrix_added': 0,
            'l1_replaced': 0,
            'skipped': 0
        }

        for img_path in image_files:
            img_name = img_path.stem
            label_path = Path(labels_dir) / f"{img_name}.txt"

            if not label_path.exists():
                print(f"Warning: No label file for {img_name}, skipping...")
                stats['skipped'] += 1
                continue

            # Create augmentations
            for aug_idx in range(num_augmentations_per_image):
                try:
                    syn_image, syn_annotations = self.generate_synthetic_image(
                        str(img_path),
                        str(label_path),
                        num_barcodes_range=(2, 5),
                        replace_detected=replace_detected,
                        add_random_barcodes=add_random_barcodes
                    )

                    # Count added barcodes
                    original_anns = self.load_yolo_annotations(str(label_path))
                    code128_count = sum(1 for ann in syn_annotations if ann.class_id == self.shelf_barcode_class) - \
                                   sum(1 for ann in original_anns if ann.class_id == self.shelf_barcode_class)
                    datamatrix_count = sum(1 for ann in syn_annotations if ann.class_id == self.product_barcode_class) - \
                                      sum(1 for ann in original_anns if ann.class_id == self.product_barcode_class)

                    stats['code128_added'] += code128_count
                    stats['datamatrix_added'] += datamatrix_count

                    # Save
                    output_name = f"{img_name}_syn_{aug_idx}"
                    output_img_path = Path(output_images_dir) / f"{output_name}.jpg"
                    output_label_path = Path(output_labels_dir) / f"{output_name}.txt"

                    cv2.imwrite(str(output_img_path), syn_image)
                    self.save_yolo_annotations(syn_annotations, str(output_label_path))

                    print(f"✓ {output_name} (+{code128_count} Code128, +{datamatrix_count} DataMatrix)")
                    stats['processed'] += 1

                except Exception as e:
                    print(f"✗ Error processing {img_name}: {e}")

        print(f"\n{'='*60}")
        print(f"Generation Complete!")
        print(f"  - Processed: {stats['processed']} images")
        print(f"  - Code128 added: {stats['code128_added']}")
        print(f"  - DataMatrix added: {stats['datamatrix_added']}")
        print(f"  - Skipped: {stats['skipped']}")
        print(f"{'='*60}\n")
