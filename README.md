# sam2cpp

A C++ SDK for the Segment Anything 2 (SAM2) model, providing high-performance image and video segmentation capabilities. Built on top of the MNN (Mobile Neural Network) framework.

## Overview

sam2cpp is a C++ implementation of the Segment Anything 2 (SAM2) model, designed for efficient image and video segmentation. It provides a simple API for integrating SAM2 into C++ applications, with support for both image and video processing. The implementation leverages the MNN framework for optimized inference across different hardware platforms.

## Features

- **Image Segmentation**: Segment objects in images using points, boxes, or both
- **Video Segmentation**: Track objects across video frames
- **Multiple Backend Support**: CPU, CUDA, OpenCL, and Metal inference backends via MNN
- **Cross-Platform**: Works on Linux, macOS, and Android
- **Simple API**: Easy-to-use C++ interface
- **Demo Applications**: Sample applications for both image and video segmentation

## Supported Models

sam2cpp currently supports the following models:

1. **Segment Anything 2 (SAM2)** - [GitHub](https://github.com/facebookresearch/sam2)
   - High-quality image and video segmentation model
   - Supports point, box, and text prompts

2. **Efficient Track Anything Model (EfficientTAM)** - [GitHub](https://github.com/yformer/EfficientTAM)
   - Efficient video object tracking and segmentation
   - Optimized for real-time performance

## Model Conversion

To use these models with sam2cpp, you need to convert them from PyTorch (.pt) format to MNN format. The conversion process involves two steps:

### Step 1: Convert PyTorch to ONNX

1. For SAM2 model:
   ```bash
   python export_sam_onnx.py --checkpoint path/to/sam2_model.pt --output path/to/sam2_model
   ```

2. For EfficientTAM model:
   ```bash
   python export_etam_onnx.py --checkpoint path/to/etam_model.pt --output path/to/etam_model
   ```

### Step 2: Convert ONNX to MNN

```bash
mkdir mnn_models

MNNConvert -f ONNX --modelFile path/to/sam2_model/image_decoder.onnx --MNNModel mnn_models/image_decoder.mnn --bizCode biz

/MNNConvert -f ONNX --modelFile path/to/sam2_mode/image_encoder.onnx --MNNModel mnn_models/image_encoder.mnn --bizCode biz

MNNConvert -f ONNX --modelFile path/to/sam2_mode/memory_encoder.onnx --MNNModel mnn_models/memory_encoder.mnn --bizCode biz

MNNConvert -f ONNX --modelFile path/to/sam2_mode/memory_attention.onnx --MNNModel mnn_models/memory_attention.mnn --bizCode biz

```

The resulting MNN model files can be used directly with the sam2cpp library.

## Requirements

- C++17 compatible compiler
- CMake 3.10 or higher
- MNN (Mobile Neural Network) library
- OpenCV (for demo applications)

## Installation

### Building from Source

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/sam2cpp.git
   cd sam2cpp
   ```

2. Build the library:
   ```bash
   mkdir build
   cd build
   cmake .. -DMNN_DIR=/path/to/mnn/lib
   make
   ```

3. Install the library (optional):
   ```bash
   make install
   ```

### Building Demo Applications

To build the demo applications, you need to specify the paths to OpenCV and the installed sam2cpp library:

```bash
cd demo
mkdir build
cd build
cmake .. -DOpenCV_DIR=/data/lib/opencv/build/install/lib/cmake/opencv4 -DSAM2CPP_DIR=../build/install
make
```

## Usage

### Image Segmentation

```cpp
#include "sam2_image.h"

using namespace sam2;

// Initialize the predictor
SAM2Image predictor;
Params params;
auto init_result = predictor.initialize("path/to/model", params);
if (!init_result) {
    std::cerr << "Initialization failed: " << init_result.error().message << std::endl;
    return -1;
}

// Get image embedding
auto embedding_result = predictor.get_embedding("path/to/image.jpg");
if (!embedding_result) {
    std::cerr << "Failed to get embedding: " << embedding_result.error().message << std::endl;
    return -1;
}

// Define points and boxes
std::vector<Point> points = {{100, 100, 1}, {200, 200, 0}}; // Positive and negative points
std::vector<BBox> boxes = {{50, 50, 150, 150}}; // Bounding box

// Predict mask
auto mask_result = predictor.predict(embedding_result.value(), points, boxes);
if (!mask_result) {
    std::cerr << "Prediction failed: " << mask_result.error().message << std::endl;
    return -1;
}

// Use the mask
const Mask& mask = mask_result.value();
// mask.data contains the binary mask
// mask.height and mask.width contain the dimensions
// mask.score contains the confidence score
```

### Video Segmentation

```cpp
#include "sam2_video.h"

using namespace sam2;

// Initialize the predictor
SAM2Video predictor;
Params params;
auto init_result = predictor.initialize("path/to/model", params);
if (!init_result) {
    std::cerr << "Initialization failed: " << init_result.error().message << std::endl;
    return -1;
}

// Create inference state
InferenceState inference_state;

// Add points or boxes for tracking
std::vector<Point> points = {{100, 100, 1}};
std::vector<BBox> boxes = {};
auto result = predictor.add_new_points_or_box(
    inference_state,
    "path/to/frame.jpg",
    1, // object ID
    points,
    boxes
);

// Track objects in subsequent frames
auto track_result = predictor.track_step(inference_state, "path/to/next_frame.jpg");
if (!track_result) {
    std::cerr << "Tracking failed: " << track_result.error().message << std::endl;
    return -1;
}

// Get tracking results
auto [frame_idx, outputs] = track_result.value();
// outputs is a map of object IDs to masks
```

## Demo Applications

The repository includes two demo applications:

1. **Image Demo**: Interactive application for image segmentation
   ```bash
   # ./bin/sam2_image_demo <model_path> <image_path>
   ./bin/sam2_image_demo ../../mnn_models ../../images/truck.jpg
   ```

2. **Video Demo**: Application for video segmentation
   ```bash
   # ./bin/sam2_video_demo <model_path> <frames_directory> <output_directory> [max_frames]

   ./bin/sam2_video_demo ../../mnn_models ../../images/bedroom ./outputs 10
   ```

## Acknowledgments

- This project is based on the Segment Anything 2 (SAM2) model
- Uses MNN (Mobile Neural Network) for efficient inference