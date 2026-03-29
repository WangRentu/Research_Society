#!/bin/bash
# Build the Cogito superimage for AIRS-Bench
# Usage: bash superimage/build.sh
#
# Requires: apptainer (install via conda: conda install -c conda-forge apptainer)
# Output: shared/superimage/superimage.root.2025-05-02v2.sif

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
OUTPUT_DIR="$PROJECT_DIR/shared/superimage"
VERSION="2025-05-02v2"
SIF_NAME="superimage.root.${VERSION}.sif"

mkdir -p "$OUTPUT_DIR"

# Use /data1 for temp (root partition too small)
export APPTAINER_TMPDIR="/data1/intern/tmp/apptainer_build"
export TMPDIR="/data1/intern/tmp/apptainer_build"
mkdir -p "$APPTAINER_TMPDIR"

echo "================================================"
echo "  Building Cogito Superimage"
echo "  Def:    $SCRIPT_DIR/apptainer.def"
echo "  Output: $OUTPUT_DIR/$SIF_NAME"
echo "  Tmpdir: $APPTAINER_TMPDIR"
echo "================================================"

cd "$SCRIPT_DIR"

apptainer build --fakeroot --tmpdir "$APPTAINER_TMPDIR" \
    "$OUTPUT_DIR/$SIF_NAME" \
    apptainer.def

echo ""
echo "================================================"
echo "  Build complete!"
echo "  Image: $OUTPUT_DIR/$SIF_NAME"
echo "  Size:  $(du -h "$OUTPUT_DIR/$SIF_NAME" | cut -f1)"
echo ""
echo "  To use: set SUPERIMAGE_DIR=$OUTPUT_DIR"
echo "  in .env or export it before running experiments"
echo "================================================"
