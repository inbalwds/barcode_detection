"""
Synthetic Barcode Data Generator
Generates Code128 and DataMatrix barcodes with augmentations
"""

import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
import random
import string
from typing import Tuple, Dict, List
from dataclasses import dataclass

try:
    import barcode
    from barcode.writer import ImageWriter
except ImportError:
    print("Installing python-barcode...")
    import subprocess
    subprocess.check_call(['pip', 'install', 'python-barcode', '--break-system-packages'])
    import barcode
    from barcode.writer import ImageWriter

# DataMatrix support - commented out for now (requires libdmtx system library)
# We'll focus on Code128 first
# try:
#     from pylibdmtx.pylibdmtx import encode as dm_encode
# except ImportError:
#     pass


@dataclass
class BarcodeAnnotation:
    """Annotation for a single barcode"""
    bbox: Tuple[int, int, int, int]  # x, y, w, h
    barcode_type: str  # 'code128' or 'datamatrix'
    text: str  # The encoded text
    

class BarcodeGenerator:
    """Generate synthetic barcode images with annotations"""
    
    def __init__(self, image_size: Tuple[int, int] = (640, 480)):
        self.image_size = image_size
        # Starting with Code128 only - easier to debug
        self.barcode_types = ['code128']
        
    def generate_random_text(self, barcode_type: str = 'code128', min_len: int = 5, max_len: int = 20) -> str:
        """Generate random text for barcode"""
        length = random.randint(min_len, max_len)
        # Code128 supports ASCII characters
        chars = string.ascii_uppercase + string.digits
        return ''.join(random.choices(chars, k=length))
    
    def create_code128(self, text: str) -> np.ndarray:
        """Create Code128 barcode image"""
        # Create barcode
        code128 = barcode.get_barcode_class('code128')
        
        # Generate with custom options
        writer = ImageWriter()
        writer.dpi = 300
        
        # Customize appearance
        options = {
            'module_width': 0.3,
            'module_height': 10.0,
            'quiet_zone': 2.0,
            'font_size': 0,  # No text below barcode
            'text_distance': 1.0,
            'background': 'white',
            'foreground': 'black',
        }
        
        # Create barcode
        barcode_obj = code128(text, writer=writer)
        
        # Render to image
        import io
        buffer = io.BytesIO()
        barcode_obj.write(buffer, options=options)
        buffer.seek(0)
        
        # Convert to numpy array
        pil_image = Image.open(buffer)
        img_array = np.array(pil_image.convert('RGB'))
        
        return img_array
    
    def create_datamatrix(self, text: str) -> np.ndarray:
        """Create DataMatrix barcode image"""
        # Encode text to DataMatrix
        encoded = dm_encode(text.encode('utf-8'))
        
        # Create image from DataMatrix
        # The encoded object contains the barcode as a 2D array
        img = Image.frombytes('RGB', (encoded.width, encoded.height), encoded.pixels)
        
        # Scale up for better visibility
        scale_factor = 5
        new_size = (encoded.width * scale_factor, encoded.height * scale_factor)
        img = img.resize(new_size, Image.NEAREST)
        
        return np.array(img)
    
    def apply_augmentations(self, barcode_img: np.ndarray, 
                          severity: str = 'medium') -> np.ndarray:
        """Apply random augmentations to barcode"""
        img = barcode_img.copy()
        
        if severity == 'light':
            aug_prob = 0.3
        elif severity == 'medium':
            aug_prob = 0.5
        else:  # heavy
            aug_prob = 0.7
        
        # 1. Rotation
        if random.random() < aug_prob:
            angle = random.uniform(-30, 30)
            h, w = img.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            img = cv2.warpAffine(img, M, (w, h), 
                                borderMode=cv2.BORDER_CONSTANT,
                                borderValue=(255, 255, 255))
        
        # 2. Perspective transform
        if random.random() < aug_prob * 0.5:
            h, w = img.shape[:2]
            pts1 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
            offset = int(min(w, h) * 0.1)
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
        if random.random() < aug_prob:
            kernel_size = random.choice([3, 5])
            img = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
        
        # 4. Noise
        if random.random() < aug_prob:
            noise = np.random.normal(0, 10, img.shape).astype(np.uint8)
            img = cv2.add(img, noise)
        
        # 5. Brightness/Contrast
        if random.random() < aug_prob:
            alpha = random.uniform(0.7, 1.3)  # Contrast
            beta = random.randint(-30, 30)    # Brightness
            img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
        
        # 6. Partial occlusion (simulate damaged barcode)
        if random.random() < aug_prob * 0.3:
            h, w = img.shape[:2]
            x1 = random.randint(0, w - w//4)
            y1 = random.randint(0, h - h//4)
            x2 = x1 + random.randint(w//10, w//4)
            y2 = y1 + random.randint(h//10, h//4)
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), -1)
        
        return img
    
    def place_barcode_on_background(self,
                                    num_barcodes: int = 1) -> Tuple[np.ndarray, List[BarcodeAnnotation]]:
        """Place barcode(s) on a background image"""
        # Create background
        background = np.ones((self.image_size[1], self.image_size[0], 3), dtype=np.uint8) * 255
        
        # Add texture to background
        noise = np.random.normal(240, 10, background.shape).astype(np.uint8)
        background = cv2.addWeighted(background, 0.7, noise, 0.3, 0)
        
        annotations = []
        
        for _ in range(num_barcodes):
            # Using Code128 for now
            barcode_type = 'code128'
            text = self.generate_random_text(barcode_type)
            
            # Generate barcode
            barcode = self.create_code128(text)
            
            # Apply augmentations
            barcode = self.apply_augmentations(barcode, severity='medium')
            
            # Resize barcode randomly
            scale = random.uniform(0.5, 1.5)
            new_h = int(barcode.shape[0] * scale)
            new_w = int(barcode.shape[1] * scale)
            
            # Make sure it fits in the background
            new_h = min(new_h, self.image_size[1] - 20)
            new_w = min(new_w, self.image_size[0] - 20)
            
            barcode = cv2.resize(barcode, (new_w, new_h))
            
            # Random position
            max_y = self.image_size[1] - new_h
            max_x = self.image_size[0] - new_w
            
            if max_y <= 0 or max_x <= 0:
                continue
            
            y = random.randint(0, max_y)
            x = random.randint(0, max_x)
            
            # Place barcode on background
            background[y:y+new_h, x:x+new_w] = barcode
            
            # Create annotation
            annotation = BarcodeAnnotation(
                bbox=(x, y, new_w, new_h),
                barcode_type=barcode_type,
                text=text
            )
            annotations.append(annotation)
        
        return background, annotations
    
    def generate_batch(self, batch_size: int = 8, 
                      max_barcodes_per_image: int = 3) -> List[Tuple[np.ndarray, List[BarcodeAnnotation]]]:
        """Generate a batch of images with barcodes"""
        batch = []
        
        for _ in range(batch_size):
            num_barcodes = random.randint(1, max_barcodes_per_image)
            img, annotations = self.place_barcode_on_background(num_barcodes=num_barcodes)
            batch.append((img, annotations))
        
        return batch
    
    def visualize_annotations(self, img: np.ndarray, 
                            annotations: List[BarcodeAnnotation]) -> np.ndarray:
        """Draw bounding boxes and labels on image"""
        vis_img = img.copy()
        
        for ann in annotations:
            x, y, w, h = ann.bbox
            
            # Draw bbox
            color = (0, 255, 0) if ann.barcode_type == 'code128' else (0, 0, 255)
            cv2.rectangle(vis_img, (x, y), (x+w, y+h), color, 2)
            
            # Draw label
            label = f"{ann.barcode_type}: {ann.text[:10]}..."
            cv2.putText(vis_img, label, (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return vis_img


if __name__ == "__main__":
    # Test the generator
    print("Creating barcode generator...")
    generator = BarcodeGenerator(image_size=(640, 480))
    
    print("Generating sample images...")
    batch = generator.generate_batch(batch_size=5, max_barcodes_per_image=2)
    
    # Save some samples
    for i, (img, annotations) in enumerate(batch):
        # Visualize
        vis_img = generator.visualize_annotations(img, annotations)
        
        # Save
        cv2.imwrite(f'sample_{i+1}.png', vis_img)
        
        # Print annotations
        print(f"\nImage {i+1}:")
        for j, ann in enumerate(annotations):
            print(f"  Barcode {j+1}: {ann.barcode_type} = '{ann.text}'")
            print(f"    BBox: {ann.bbox}")
    
    print("\n✅ Done! Check sample_*.png files")