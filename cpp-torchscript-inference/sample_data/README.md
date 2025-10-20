# Sample Data

This directory contains sample models and images for testing the inference engines.

## Models

- **nails_63.5MAP.torchscript.pt** (36MB): Nail detection model with 63.5% MAP performance
  - Input size: 960×960 pixels
  - Output format: Bounding boxes with class IDs and confidence scores
  - Classes: Various nail defect types (47 classes total)

## Images

- **test.jpg**: Test image for single inference testing
- **output_1754688062.jpg**: Additional test image for batch processing

## Usage Examples

```bash
# Single image inference
./run_inference.sh simple sample_data/models/nails_63.5MAP.torchscript.pt sample_data/images/test.jpg

# Batch processing
./run_inference.sh batch sample_data/models/nails_63.5MAP.torchscript.pt sample_data/images/
```

## Adding Your Own Data

To use your own models and images:

1. **Models**: Place TorchScript `.pt` files in `sample_data/models/` or any directory
2. **Images**: Add images in common formats (JPG, PNG, BMP) to `sample_data/images/` or any directory

The inference engines automatically detect supported file formats and process them accordingly.