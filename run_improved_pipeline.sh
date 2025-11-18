#!/bin/bash

echo "=========================================="
echo "Improved Synthetic Barcode Generator v2.0"
echo "=========================================="
echo ""
echo "Features:"
echo "  ✓ Code128 on shelves (realistic sizes)"
echo "  ✓ DataMatrix on pallets (realistic sizes)"
echo "  ✓ Sizes match your training data distribution"
echo ""
echo "This will process ALL 6,326 warehouse images"
echo "and create ~19,000 synthetic images with realistic barcodes."
echo ""
echo "Estimated time: 1-2 hours"
echo ""
read -p "Continue? (y/n) " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]
then
    echo "Starting improved pipeline..."
    python improved_pipeline.py
else
    echo "Cancelled."
    echo ""
    echo "To test on 5 images first, run:"
    echo "  python test_improved.py"
fi
