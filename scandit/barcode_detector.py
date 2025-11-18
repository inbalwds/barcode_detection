#!/usr/bin/env python3
import os, sys, re, csv, fnmatch, time, yaml, tempfile
import numpy as np
import cv2
from utils import *

# ---------- project paths ----------
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))

# Add project root and scandit directory to path
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Try to add report_generator_v2 as alternative
ALT_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", "..", "report_generator_v2"))
if os.path.exists(ALT_ROOT) and ALT_ROOT not in sys.path:
    sys.path.insert(0, ALT_ROOT)

try:
    from interfaces import ExitCode
except Exception:
    class ExitCode:
        LICENSE_ERROR = type("E", (), {"value": 2})
        OK            = type("E", (), {"value": 0})

# Try to import scandit from multiple locations
sc = None
try:
    # First try: local scandit directory
    from scandit import scanditsdk as sc
except Exception:
    try:
        # Second try: utils.scandit (report_generator_v2 style)
        from utils.scandit import scanditsdk as sc
    except Exception as e:
        print("Could not import Scandit SDK wrapper.")
        print("Import error:", e)
        print("Tried:")
        print("  1. scandit.scanditsdk")
        print("  2. utils.scandit.scanditsdk")
        sys.exit(1)

# ===========================
# Scandit Decoder
# ===========================
class ScanditDecoder:
    def __init__(self, license_key):
        try:
            self.context  = sc.RecognitionContext(license_key, tempfile.gettempdir())
            self.settings = self._create_settings()
            self.scanner  = sc.BarcodeScanner(self.context, self.settings)
            self.scanner.wait_for_setup_completed()
        except Exception as e:
            print(f"Scandit initialization failed: {e}")
            sys.exit(ExitCode.LICENSE_ERROR.value)

    def _create_settings(self):
        # Use PRESET_NONE for maximum control (from original working code)
        s = sc.BarcodeScannerSettings(preset=sc.PRESET_NONE)
        s.focus_mode = sc.CAMERA_FOCUS_MODE_FIXED
        s.max_number_of_codes_per_frame = 20
        s.code_location_constraint_1d = sc.CODE_LOCATION_IGNORE
        s.code_location_constraint_2d = sc.CODE_LOCATION_IGNORE
        s.code_direction_hint = sc.CODE_DIRECTION_NONE

        for sym in [sc.SYMBOLOGY_DATA_MATRIX, sc.SYMBOLOGY_EAN13_UPCA,
                    sc.SYMBOLOGY_QR, sc.SYMBOLOGY_CODE128]:
            s.enable_symbology(sym, True)

        # Enable settings as per original working code
        s.symbologies[sc.SYMBOLOGY_EAN13_UPCA].color_inverted_enabled = True
        dm = s.symbologies[sc.SYMBOLOGY_DATA_MATRIX]
        dm.set_extension_enabled("direct_part_marking_mode", True)
        # Note: Not enabling color_inverted for DataMatrix as in original code

        return s

    # ---------- image prep & description ----------
    def _to_8bit_rgb_or_gray(self, image):
        # keep 8-bit; convert 16-bit -> 8-bit
        if image.dtype == np.uint16:
            image = cv2.convertScaleAbs(image, alpha=255.0/65535.0)

        if image.ndim == 2:
            work = image if image.dtype == np.uint8 else image.astype(np.uint8)
        else:
            # cv2 loads BGR; Scandit expects RGB when layout is RGB_8U
            work = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if not work.flags["C_CONTIGUOUS"]:
            work = np.ascontiguousarray(work)
        return work

    def _maybe_downscale(self, img, max_side=2048):
        h, w = img.shape[:2]
        if max(h, w) <= max_side:
            return img
        scale = max_side / float(max(h, w))
        return cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)

    def _get_descr(self, img8):
        d = sc.ImageDescription()
        h, w = img8.shape[:2]
        d.width  = w
        d.height = h
        d.layout = sc.IMAGE_LAYOUT_GRAY_8U if img8.ndim == 2 else sc.IMAGE_LAYOUT_RGB_8U
        d.set_plane_row_bytes(0, img8.strides[0])  # real row stride in bytes
        return d

    # ---------- try multiple variants until success ----------
    def _variants(self, base):
        """Yield a few robust variants (order chosen to be fast→strong)."""
        out = []

        # 0) original (8-bit gray or RGB) and optional downscale
        b0 = self._maybe_downscale(base, 2048)
        out.append(("base", b0))

        # 1) CLAHE on gray
        if b0.ndim == 3:
            gray = cv2.cvtColor(b0, cv2.COLOR_RGB2GRAY)
        else:
            gray = b0
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        out.append(("clahe", clahe.apply(gray)))

        # 2) mild blur (helps noisy DM) on gray
        out.append(("blur", cv2.GaussianBlur(gray, (3, 3), 0)))

        # 3) adaptive threshold (and its inverse)
        th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 31, 5)
        out.append(("thresh", th))
        out.append(("thresh_inv", 255 - th))

        # 4) tiny rotations (±3°, ±7°)
        for ang in (-7, -3, 3, 7):
            M = cv2.getRotationMatrix2D((gray.shape[1]//2, gray.shape[0]//2), ang, 1.0)
            rot = cv2.warpAffine(gray, M, (gray.shape[1], gray.shape[0]),
                                 flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            out.append((f"rot{ang}", rot))

        return out

    def _process_once(self, img8):
        """Process a single 8-bit image (gray or RGB)."""
        frame_seq = self.context.start_new_frame_sequence()
        descr = self._get_descr(img8)
        ptr = img8.__array_interface__['data'][0]
        result = frame_seq.process_frame(descr, ptr)
        frame_seq.end()

        if result.status != sc.RECOGNITION_CONTEXT_STATUS_SUCCESS:
            return []

        hits = []
        for code in self.scanner.session.newly_recognized_codes:
            q = code.location
            hits.append({
                "Decoded Data": code.data,
                "Symbology": code.symbology_string,
                "Corners": [
                    (q.top_left.x, q.top_left.y),
                    (q.top_right.x, q.top_right.y),
                    (q.bottom_right.x, q.bottom_right.y),
                    (q.bottom_left.x, q.bottom_left.y),
                ],
                "detection_method": "Scandit_Enhanced_Config"
            })
        return hits

    def decode_image(self, image):
        """Try multiple pre-processing variants; return on first success."""
        base = self._to_8bit_rgb_or_gray(image)

        # IMPORTANT: some variants are gray; some are RGB (base may be RGB).
        # We clear Scandit's internal list by starting a new frame sequence each call,
        # so no explicit reset_session() is required (and not available in your wrapper).

        # First try the base (fast path)
        hits = self._process_once(base)
        if hits:
            return hits

        # Then try gray-based variants
        for name, var in self._variants(base):
            # ensure 8-bit
            if var.dtype != np.uint8:
                var = var.astype(np.uint8)
            if not var.flags["C_CONTIGUOUS"]:
                var = np.ascontiguousarray(var)
            hits = self._process_once(var)
            if hits:
                return hits
        return []
    
    def decode_and_mark(self, image_path, decoder, marked_dir, decoded_dict):
        img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            return [], 0.0

        t0 = time.perf_counter()
        results = decoder.decode_image(img)
        dt = time.perf_counter() - t0

        vis = img if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        for r in results:
            data, corners = r["Decoded Data"], r["Corners"]
            decoded_dict.setdefault(data, []).append({"image_name": os.path.basename(image_path),
                                                    "corners": corners})
            cv2.polylines(vis, [np.array(corners, np.int32)], True, (0,255,0), 2)
            cv2.putText(vis, data, corners[0], cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,0), 2)

        os.makedirs(marked_dir, exist_ok=True)
        cv2.imwrite(os.path.join(marked_dir, os.path.basename(image_path)), vis)
        print(f"Image: {os.path.basename(image_path):30} | Barcodes: {len(results):2} | Decode time: {dt*1000:6.2f}ms")
        return results, dt
