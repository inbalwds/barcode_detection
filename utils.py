import yaml
import os
import fnmatch
import re
import csv

def load_config(script_dir):
    p = os.path.join(script_dir, "config.yaml")
    if os.path.exists(p):
        with open(p, "r") as f:
            return yaml.safe_load(f)
    return {}

def extract_number(fn):
    m = re.search(r'_(\d+)', fn)
    return int(m.group(1)) if m else 0

def get_files_list_from_directory(directory, patterns):
    out = []
    for root, _, files in os.walk(directory):
        for f in files:
            if any(fnmatch.fnmatch(f, pat) for pat in patterns):
                out.append(os.path.join(root, f))
    out.sort(key=lambda x: extract_number(os.path.basename(x)))
    return out

def write_barcodes_csv(decoded, out_csv):
    product_pat = re.compile(r'^L1(0{9}\d{9})$')
    shelf_pat   = re.compile(r'^(?P<passage>\d{2})(?P<col>\d{2})(?P<level>\d{1})(?P<sub_col>\d{2})$')
    pc = sc_ = oc = 0
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, quoting=csv.QUOTE_ALL, escapechar='\\')
        w.writerow(["Decoded Data", "Image Filename", "Corners", "Type"])
        for data, items in decoded.items():
            if product_pat.match(data):
                tp = "product"; pc += len(items)
            elif shelf_pat.match(data):
                tp = "shelf";   sc_ += len(items)
            else:
                tp = "other";   oc  += len(items)
            for it in items:
                cstr = "-".join([f"({x},{y})" for x, y in it["corners"]])
                w.writerow([data, it["image_name"], cstr, tp])
    print(f"Summary:\n  Product barcodes: {pc}\n  Shelf barcodes:   {sc_}\n  Other barcodes:   {oc}")

def write_timing_csv(rows, out_csv):
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, quoting=csv.QUOTE_ALL, escapechar='\\')
        w.writerow(["Image Filename", "Decode Time (ms)", "Barcodes Found", "Time per Barcode (ms)"])
        for r in rows:
            ms = r["decode_time"] * 1000.0
            cnt = r["barcodes_count"]
            w.writerow([r["filename"], f"{ms:.2f}", cnt, f"{(ms/max(1,cnt)):.2f}"])

