#!/bin/bash

# Complete pipeline to generate synthetic barcode dataset
# Usage: bash run_full_pipeline.sh

echo "=========================================="
echo "Synthetic Barcode Dataset Generator"
echo "=========================================="
echo ""
echo "This will process ALL images from your warehouse dataset"
echo "and create synthetic barcodes on detected pallets/shelves."
echo ""
echo "Input: rami levy.v16i.yolov12/train/images (6,326 images)"
echo "Output: synthetic_dataset/ (3 versions per image = ~19,000 images)"
echo ""
echo "Estimated time: 1-2 hours"
echo ""
read -p "Continue? (y/n) " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]
then
    echo "Starting pipeline..."
    python detect_and_generate_dataset.py
else
    echo "Cancelled."
    echo ""
    echo "To test on 5 images first, run:"
    echo "  python quick_test.py"
fi
