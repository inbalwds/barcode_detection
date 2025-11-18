# Barcode Detection & Synthetic Data Generator

מערכת לזיהוי ברקודים ויצירת נתונים סינטטיים לאימון מודלים של YOLO.

## תכונות עיקריות

### 🔍 זיהוי ברקודים עם Scandit
- זיהוי **DataMatrix** (כולל L1-prefix)
- זיהוי **Code128** (ברקודים על מדפים)
- תמיכה במספר וריאנטים של תמונה (CLAHE, blur, threshold, rotation)

### 🎨 יצירת נתונים סינטטיים
- **מצב היברידי חכם**:
  - אם נמצאו ברקודים → מחליף במיקום המדויק
  - אם לא נמצאו → יוצר ברקודים רנדומליים על משטחים ומדפים
- שימור orientation והגיאומטריה המקורית (perspective transform)
- ברקודים L1-prefixed אוטומטיים
- אוגמנטציות ריאליסטיות (blur, noise, rotation, lighting)

## דרישות מקדימות

```bash
pip install opencv-python numpy pillow python-barcode pyyaml
```

### Scandit SDK
- נדרש רשיון Scandit תקף
- הספרייה צריכה להיות ב-`scandit/` directory

## הגדרה מהירה

1. **הגדר את ה-config.yaml**:
```yaml
license_key: "YOUR_SCANDIT_LICENSE_KEY"
INPUT_IMAGES_DIR: '/path/to/images'
CLASS_NAMES: ['box', 'empty_shelves', 'palletes', 'product_barcodes', 'shelf', 'shelf_barcodes']
```

2. **הרץ טסט מהיר**:
```bash
python3 quick_test_scandit.py
```

3. **צור dataset מלא**:
```bash
python3 generate_synthetic_with_scandit.py \
    --input-images /path/to/images \
    --output-images /path/to/output/images \
    --num-augmentations 5
```

## שימוש פרוגרמטי

```python
from data.improved_warehouse_generator import ImprovedWarehouseBarcodeGenerator

# אתחול
generator = ImprovedWarehouseBarcodeGenerator(
    class_mapping={
        'box': 0, 'empty_shelves': 1, 'palletes': 2,
        'product_barcodes': 3, 'shelf': 4, 'shelf_barcodes': 5
    },
    scandit_license_key=license_key,
    use_scandit=True
)

# עיבוד תמונה בודדת
syn_image, annotations = generator.generate_synthetic_image(
    image_path='warehouse.jpg',
    annotation_path='warehouse.txt',
    replace_detected=True,      # זיהוי והחלפה במיקום מדויק
    add_random_barcodes=True    # הוספת ברקודים רנדומליים
)

# שמירה
import cv2
cv2.imwrite('output.jpg', syn_image)
generator.save_yolo_annotations(annotations, 'output.txt')
```

## מבנה הפרויקט

```
barcode_detection/
├── data/
│   └── improved_warehouse_generator.py  # גנרטור ראשי
├── scandit/
│   ├── barcode_detector.py             # Scandit wrapper
│   └── scanditsdk.py                   # Scandit SDK
├── quick_test_scandit.py               # טסט מהיר
├── test_scandit_synthetic_generator.py # טסט מלא
├── generate_synthetic_with_scandit.py  # סקריפט ייצור
├── config.yaml                         # הגדרות
├── .gitignore                          # Git ignore
└── README.md                           # המסמך הזה
```

## סקריפטים זמינים

### `quick_test_scandit.py`
טסט מהיר על תמונה אחת
```bash
python3 quick_test_scandit.py
```

### `test_scandit_synthetic_generator.py`
טסט על dataset קטן
```bash
python3 test_scandit_synthetic_generator.py
```

### `generate_synthetic_with_scandit.py`
ייצור dataset מלא עם אופציות CLI
```bash
python3 generate_synthetic_with_scandit.py \
    --input-images /path/to/images \
    --input-labels /path/to/labels \
    --output-images /path/to/output/images \
    --output-labels /path/to/output/labels \
    --num-augmentations 5
```

## איך זה עובד?

### 1. זיהוי ברקודים
```python
detected_barcodes = detect_all_barcodes(image)
# Returns:
# {
#   'datamatrix_l1': [{'data': 'L1ABC123', 'corners': [...]}],
#   'code128': [{'data': 'SHELF001', 'corners': [...]}],
#   'other': [...]
# }
```

### 2. החלפה במיקום מדויק
- מקבל את הקואורדינטות של הפינות
- יוצר ברקוד חדש עם L1 prefix
- משתמש ב-perspective transform להתאמת זווית
- מדביק במיקום המדויק של הברקוד המקורי

### 3. ברקודים רנדומליים (fallback)
אם לא נמצאו ברקודים:
- Code128 על מדפים
- DataMatrix (L1-prefix) על משטחים
- גדלים ריאליסטיים לפי התפלגות אמיתית

## פורמט YOLO

תוויות נוצרות בפורמט YOLO סטנדרטי:
```
class_id x_center y_center width height
5 0.4532 0.6234 0.0823 0.0512    # shelf_barcode
3 0.7821 0.3421 0.0612 0.0923    # product_barcode (DataMatrix)
```

## בעיות נפוצות

### "Could not import Scandit SDK"
וודא ש-`scandit/scanditsdk.py` קיים ו-libscandit.so נמצא

### "License validation failed"
בדוק שה-license key ב-config.yaml תקף

### "No barcodes detected"
- נסה תמונות ברזולוציה גבוהה יותר
- וודא שהברקודים ברורים ולא מטושטשים
- המערכת תפעל במצב fallback (ברקודים רנדומליים)

## תיעוד נוסף

- [SCANDIT_SYNTHETIC_DATA_README.md](SCANDIT_SYNTHETIC_DATA_README.md) - תיעוד מפורט של הזיהוי
- [UPDATED_SCANDIT_INTEGRATION.md](UPDATED_SCANDIT_INTEGRATION.md) - עדכון האינטגרציה

## רישיון

הקוד משתמש ב-Scandit SDK שדורש רשיון מסחרי.

---

**נוצר עם ❤️ על ידי Claude Code**
