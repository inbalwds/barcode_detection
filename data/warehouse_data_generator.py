"""
Warehouse Synthetic Barcode Data Generator
Adds synthetic barcodes to real warehouse images
"""

import numpy as np
import cv2
from PIL import Image
import random
import string
import os
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


@dataclass
class YOLOAnnotation:
    """YOLO format annotation"""
    class_id: int
    x_center: float  # normalized 0-1
    y_center: float  # normalized 0-1
    width: float     # normalized 0-1
    height: float    # normalized 0-1
    
    def to_bbox(self, img_width: int, img_height: int) -> Tuple[int, int, int, int]:
        """Convert YOLO format to absolute bbox (x, y, w, h)"""
        x = int((self.x_center - self.width / 2) * img_width)
        y = int((self.y_center - self.height / 2) * img_height)
        w = int(self.width * img_width)
        h = int(self.height * img_height)
        return (x, y, w, h)
    
    @staticmethod
    def from_bbox(class_id: int, bbox: Tuple[int, int, int, int], 
                  img_width: int, img_height: int) -> 'YOLOAnnotation':
        """Create YOLO annotation from absolute bbox"""
        x, y, w, h = bbox
        x_center = (x + w / 2) / img_width
        y_center = (y + h / 2) / img_height
        width = w / img_width
        height = h / img_height
        return YOLOAnnotation(class_id, x_center, y_center, width, height)
    
    def to_yolo_string(self) -> str:
        """Convert to YOLO format string"""
        return f"{self.class_id} {self.x_center:.6f} {self.y_center:.6f} {self.width:.6f} {self.height:.6f}"


class WarehouseBarcodeGenerator:
    """
    Generates synthetic barcode data on real warehouse images
    """
    
    def __init__(self, 
                 class_mapping: Dict[str, int],
                 datamatrix_enabled: bool = False):
        """
        Args:
            class_mapping: Mapping of class names to IDs
                          For Rami Levi data:
                          {'box': 0, 'empty_shelves': 1, 'palletes': 2, 
                           'product_barcodes': 3, 'shelf': 4, 'shelf_barcodes': 5}
            datamatrix_enabled: Whether to include DataMatrix (requires libdmtx)
        """
        self.class_mapping = class_mapping
        self.datamatrix_enabled = datamatrix_enabled
        
        # Class IDs for placement logic
        self.box_class = class_mapping.get('box', 0)
        self.empty_shelves_class = class_mapping.get('empty_shelves', 1)
        self.pallet_class = class_mapping.get('palletes', 2)
        self.product_barcode_class = class_mapping.get('product_barcodes', 3)
        self.shelf_class = class_mapping.get('shelf', 4)
        self.shelf_barcode_class = class_mapping.get('shelf_barcodes', 5)
    
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
    
    def save_yolo_annotations(self, annotations: List[YOLOAnnotation], 
                             txt_path: str):
        """Save YOLO annotations to txt file"""
        with open(txt_path, 'w') as f:
            for ann in annotations:
                f.write(ann.to_yolo_string() + '\n')
    
    def generate_random_text(self, barcode_type: str, 
                            min_len: int = 5, max_len: int = 20) -> str:
        """Generate random text for barcode"""
        length = random.randint(min_len, max_len)
        
        if barcode_type == 'code128':
            chars = string.ascii_uppercase + string.digits
        else:  # datamatrix
            chars = string.ascii_letters + string.digits + '-._'
        
        return ''.join(random.choices(chars, k=length))
    
    def create_code128(self, text: str) -> np.ndarray:
        """Create Code128 barcode image"""
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
        
        return img_array
    
    def apply_realistic_augmentations(self, barcode_img: np.ndarray) -> np.ndarray:
        """
        Apply realistic warehouse augmentations:
        - Slight rotation
        - Perspective transform
        - Blur (as from distant camera)
        - Lighting changes
        - Noise
        """
        img = barcode_img.copy()
        
        # 1. Slight rotation (barcodes in warehouses aren't always straight)
        if random.random() < 0.6:
            angle = random.uniform(-15, 15)
            h, w = img.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            img = cv2.warpAffine(img, M, (w, h), 
                                borderMode=cv2.BORDER_CONSTANT,
                                borderValue=(255, 255, 255))
        
        # 2. Perspective (camera angle)
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
        
        # 3. Blur (distant camera / motion)
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
        
        # 4. Lighting (warehouse can be dark/bright)
        if random.random() < 0.6:
            alpha = random.uniform(0.6, 1.4)  # Contrast
            beta = random.randint(-40, 40)    # Brightness
            img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
        
        # 5. Noise (low quality camera)
        if random.random() < 0.4:
            noise = np.random.normal(0, 15, img.shape).astype(np.uint8)
            img = cv2.add(img, noise)
        
        # 6. Partial occlusion (box covering part of barcode)
        if random.random() < 0.2:
            h, w = img.shape[:2]
            x1 = random.randint(0, w - w//5)
            y1 = random.randint(0, h - h//5)
            x2 = x1 + random.randint(w//15, w//5)
            y2 = y1 + random.randint(h//15, h//5)
            # Random color (like a box)
            color = tuple(random.randint(50, 200) for _ in range(3))
            cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)
        
        return img
    
    def find_placement_region(self, 
                             roi_bbox: Tuple[int, int, int, int],
                             placement_strategy: str = 'center') -> Tuple[int, int]:
        """
        Find a point to place the barcode within ROI
        
        Args:
            roi_bbox: (x, y, w, h) of the ROI (shelf/pallet/sticker)
            placement_strategy: 'center', 'random', 'near_sticker'
        
        Returns:
            (x, y) for barcode placement
        """
        x, y, w, h = roi_bbox
        
        if placement_strategy == 'center':
            # Center of ROI
            return (x + w // 2, y + h // 2)
        
        elif placement_strategy == 'random':
            # Random position within ROI
            margin = 0.1  # 10% margin from edges
            x_offset = int(w * margin)
            y_offset = int(h * margin)
            
            px = random.randint(x + x_offset, x + w - x_offset)
            py = random.randint(y + y_offset, y + h - y_offset)
            return (px, py)
        
        elif placement_strategy == 'near_sticker':
            # Near sticker (usually top/bottom part)
            position = random.choice(['top', 'bottom', 'left', 'right'])
            
            if position == 'top':
                return (x + w // 2, y + int(h * 0.2))
            elif position == 'bottom':
                return (x + w // 2, y + int(h * 0.8))
            elif position == 'left':
                return (x + int(w * 0.2), y + h // 2)
            else:  # right
                return (x + int(w * 0.8), y + h // 2)
        
        return (x + w // 2, y + h // 2)
    
    def place_barcode_on_image(self,
                               image: np.ndarray,
                               barcode_img: np.ndarray,
                               position: Tuple[int, int],
                               scale_range: Tuple[float, float] = (0.8, 1.5),
                               blend: bool = True) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        """
        Place barcode on image at specific position
        
        Returns:
            (modified_image, bbox) - updated image and barcode bbox
        """
        img = image.copy()
        
        # Scale the barcode
        scale = random.uniform(*scale_range)
        new_h = int(barcode_img.shape[0] * scale)
        new_w = int(barcode_img.shape[1] * scale)
        barcode_resized = cv2.resize(barcode_img, (new_w, new_h))
        
        # Calculate position (position is center)
        x = position[0] - new_w // 2
        y = position[1] - new_h // 2
        
        # Ensure barcode is within image bounds
        x = max(0, min(x, img.shape[1] - new_w))
        y = max(0, min(y, img.shape[0] - new_h))
        
        # Blending (blend with background)
        if blend:
            # Create alpha mask - barcode not completely opaque
            alpha = random.uniform(0.85, 1.0)
            
            roi = img[y:y+new_h, x:x+new_w]
            blended = cv2.addWeighted(barcode_resized, alpha, roi, 1-alpha, 0)
            img[y:y+new_h, x:x+new_w] = blended
        else:
            img[y:y+new_h, x:x+new_w] = barcode_resized
        
        bbox = (x, y, new_w, new_h)
        return img, bbox
    
    def generate_synthetic_image(self,
                                image_path: str,
                                annotation_path: str,
                                num_barcodes_range: Tuple[int, int] = (2, 5),
                                add_random_barcodes: bool = True) -> Tuple[np.ndarray, List[YOLOAnnotation]]:
        """
        Generate synthetic image with barcodes
        
        Args:
            image_path: Path to warehouse image
            annotation_path: Path to YOLO annotations
            num_barcodes_range: How many barcodes to add
            add_random_barcodes: Whether to add barcodes in random locations too
        
        Returns:
            (image, annotations) - New image and its annotations
        """
        # 1. Load image and existing annotations
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Cannot load image: {image_path}")
        
        img_h, img_w = image.shape[:2]
        existing_annotations = self.load_yolo_annotations(annotation_path)
        
        # 2. Separate annotations by type
        shelves = [ann for ann in existing_annotations if ann.class_id == self.shelf_class]
        empty_shelves = [ann for ann in existing_annotations if ann.class_id == self.empty_shelves_class]
        pallets = [ann for ann in existing_annotations if ann.class_id == self.pallet_class]
        boxes = [ann for ann in existing_annotations if ann.class_id == self.box_class]
        existing_product_barcodes = [ann for ann in existing_annotations if ann.class_id == self.product_barcode_class]
        existing_shelf_barcodes = [ann for ann in existing_annotations if ann.class_id == self.shelf_barcode_class]
        
        # 3. List of barcodes we'll add
        new_annotations = existing_annotations.copy()
        num_barcodes = random.randint(*num_barcodes_range)
        
        for _ in range(num_barcodes):
            # Choose barcode type and location
            # Strategies:
            # 1. shelf_barcode on shelf (Code128)
            # 2. shelf_barcode on empty shelf (Code128)
            # 3. product_barcode on pallet (DataMatrix - if enabled)
            # 4. product_barcode on box (DataMatrix - if enabled)
            # 5. random
            
            if self.datamatrix_enabled:
                barcode_type = random.choices(
                    ['shelf_barcode_on_shelf', 'shelf_barcode_on_empty', 
                     'product_barcode_on_pallet', 'product_barcode_on_box', 'random'],
                    weights=[0.25, 0.15, 0.2, 0.2, 0.2]
                )[0]
            else:
                # Only Code128 (shelf_barcodes)
                barcode_type = random.choices(
                    ['shelf_barcode_on_shelf', 'shelf_barcode_on_empty', 'random'],
                    weights=[0.4, 0.3, 0.3]
                )[0]
            
            barcode_img = None
            position = None
            class_id = None
            
            if barcode_type == 'shelf_barcode_on_shelf' and len(shelves) > 0:
                # Code128 on shelf
                shelf = random.choice(shelves)
                shelf_bbox = shelf.to_bbox(img_w, img_h)
                
                text = self.generate_random_text('code128', min_len=8, max_len=15)
                barcode_img = self.create_code128(text)
                barcode_img = self.apply_realistic_augmentations(barcode_img)
                
                position = self.find_placement_region(shelf_bbox, 'random')
                class_id = self.shelf_barcode_class
            
            elif barcode_type == 'shelf_barcode_on_empty' and len(empty_shelves) > 0:
                # Code128 on empty shelf
                empty_shelf = random.choice(empty_shelves)
                shelf_bbox = empty_shelf.to_bbox(img_w, img_h)
                
                text = self.generate_random_text('code128', min_len=8, max_len=15)
                barcode_img = self.create_code128(text)
                barcode_img = self.apply_realistic_augmentations(barcode_img)
                
                position = self.find_placement_region(shelf_bbox, 'center')
                class_id = self.shelf_barcode_class
            
            elif barcode_type == 'product_barcode_on_pallet' and len(pallets) > 0 and self.datamatrix_enabled:
                # DataMatrix on pallet
                # TODO: Add DataMatrix generation when libdmtx is available
                continue
            
            elif barcode_type == 'product_barcode_on_box' and len(boxes) > 0 and self.datamatrix_enabled:
                # DataMatrix on box
                # TODO: Add DataMatrix generation when libdmtx is available
                continue
            
            elif barcode_type == 'random' and add_random_barcodes:
                # Barcode in random location (shelf_barcode - Code128)
                text = self.generate_random_text('code128', min_len=5, max_len=12)
                barcode_img = self.create_code128(text)
                barcode_img = self.apply_realistic_augmentations(barcode_img)
                
                position = (random.randint(50, img_w - 50), 
                           random.randint(50, img_h - 50))
                class_id = self.shelf_barcode_class
            
            else:
                continue
            
            # Add barcode to image
            if barcode_img is not None and position is not None:
                image, bbox = self.place_barcode_on_image(
                    image, barcode_img, position,
                    scale_range=(0.5, 1.2),
                    blend=True
                )
                
                # Create new annotation
                yolo_ann = YOLOAnnotation.from_bbox(
                    class_id, bbox, img_w, img_h
                )
                new_annotations.append(yolo_ann)
        
        return image, new_annotations
    
    def process_dataset(self,
                       images_dir: str,
                       labels_dir: str,
                       output_images_dir: str,
                       output_labels_dir: str,
                       num_augmentations_per_image: int = 3):
        """
        Process entire dataset - create synthetic augmentations for all images
        
        Args:
            images_dir: Input images directory
            labels_dir: Input labels directory (YOLO format)
            output_images_dir: Output images directory
            output_labels_dir: Output labels directory
            num_augmentations_per_image: How many versions to create per image
        """
        os.makedirs(output_images_dir, exist_ok=True)
        os.makedirs(output_labels_dir, exist_ok=True)
        
        image_files = list(Path(images_dir).glob('*.jpg')) + \
                     list(Path(images_dir).glob('*.png'))
        
        print(f"Processing {len(image_files)} images...")
        
        for img_path in image_files:
            # Paths
            img_name = img_path.stem
            label_path = Path(labels_dir) / f"{img_name}.txt"
            
            if not label_path.exists():
                print(f"Warning: No label file for {img_name}, skipping...")
                continue
            
            # Create augmentations
            for aug_idx in range(num_augmentations_per_image):
                try:
                    syn_image, syn_annotations = self.generate_synthetic_image(
                        str(img_path),
                        str(label_path),
                        num_barcodes_range=(2, 5),
                        add_random_barcodes=True
                    )
                    
                    # Save
                    output_name = f"{img_name}_syn_{aug_idx}"
                    output_img_path = Path(output_images_dir) / f"{output_name}.jpg"
                    output_label_path = Path(output_labels_dir) / f"{output_name}.txt"
                    
                    cv2.imwrite(str(output_img_path), syn_image)
                    self.save_yolo_annotations(syn_annotations, str(output_label_path))
                    
                    print(f"✓ Created {output_name}")
                
                except Exception as e:
                    print(f"✗ Error processing {img_name}: {e}")
        
        print(f"\n✅ Done! Created {len(image_files) * num_augmentations_per_image} synthetic images")


# Example usage
if __name__ == "__main__":
    # Define class mapping - exact match for Rami Levi data
    class_mapping = {
        'box': 0,
        'empty_shelves': 1,
        'palletes': 2,           # Pallets
        'product_barcodes': 3,   # Product barcodes (DataMatrix)
        'shelf': 4,
        'shelf_barcodes': 5      # Shelf barcodes (Code128)
    }
    
    # Create generator
    generator = WarehouseBarcodeGenerator(
        class_mapping=class_mapping,
        datamatrix_enabled=False  # Currently only Code128
    )
    
    # Example: Processing single image
    print("Example: Processing single image...")
    print("\nTo process your dataset, use:")
    print("""
    generator.process_dataset(
        images_dir='path/to/images',
        labels_dir='path/to/labels',
        output_images_dir='path/to/output/images',
        output_labels_dir='path/to/output/labels',
        num_augmentations_per_image=3
    )
    """)