# Player Re-Identification System

A Python-based system for detecting, tracking, and re-identifying players in sports footage. This project was developed as part of the Liat.ai AI Intern assignment.

## Overview

This system processes sports videos to:
- Detect players using YOLOv11-based object detection
- Track players across frames with unique IDs
- Re-identify players who exit and re-enter the frame
- Generate visualizations and analysis reports

## Features

- **Player Detection**: Uses YOLOv11 model for accurate player detection
- **Multi-Object Tracking**: Tracks multiple players simultaneously
- **Re-Identification**: Maintains consistent IDs when players re-enter the frame
- **Visualization**: Overlays bounding boxes and IDs on output video
- **Analysis**: Generates CSV/JSON results and statistical plots
- **CLI Interface**: Easy-to-use command-line interface

## Project Structure

```
player-reid-assignment/
├── data/                   # Input video files
├── models/                 # YOLOv11 model files
├── src/                    # Source code
│   ├── __init__.py
│   ├── detector.py         # Player detection module
│   ├── tracker.py          # Player tracking module
│   ├── visualizer.py       # Visualization module
│   └── pipeline.py         # Main pipeline
├── output/                 # Output videos
├── docs/                   # Documentation
├── main.py                 # CLI interface
├── test_simple.py          # Simple test script
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (optional, for faster processing)

### Setup

1. **Clone or download the project**:
   ```bash
   git clone <repository-url>
   cd player-reid-assignment
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Download the YOLOv11 model**:
   - Download the model from: [YOLOv11 Model Link](https://drive.google.com/file/d/1-5fOSHOSB9UxyP_enOOZNAMScrePvCMD/view)
   - Place it in the `models/` directory

5. **Download the test video**:
   - Download `15sec_input_720p.mp4` from the assignment materials
   - Place it in the `data/` directory

## Usage

### Basic Usage

```bash
python main.py --input data/15sec_input_720p.mp4 --model models/yolov11.pt --output output/result.mp4
```

### Advanced Usage

```bash
python main.py \
    --input data/15sec_input_720p.mp4 \
    --model models/yolov11.pt \
    --output output/result.mp4 \
    --conf 0.6 \
    --track-iou 0.4 \
    --feature-threshold 0.8 \
    --show-plots \
    --results results/
```

### Command Line Arguments

| Argument | Short | Required | Default | Description |
|----------|-------|----------|---------|-------------|
| `--input` | `-i` | Yes | - | Input video file path |
| `--model` | `-m` | Yes | - | YOLOv11 model file path |
| `--output` | `-o` | Yes | - | Output video file path |
| `--conf` | - | No | 0.5 | Detection confidence threshold |
| `--iou` | - | No | 0.45 | Detection IoU threshold |
| `--track-iou` | - | No | 0.3 | Tracking IoU threshold |
| `--feature-threshold` | - | No | 0.7 | Feature similarity threshold |
| `--max-missed` | - | No | 30 | Max frames to keep inactive tracks |
| `--results` | `-r` | No | results | Results output directory |
| `--show-plots` | - | No | False | Generate analysis plots |
| `--no-progress` | - | No | False | Disable progress bar |

### Testing

Run the simple test script to verify your installation:

```bash
python test_simple.py
```

## Output Files

The system generates several output files:

1. **Output Video** (`output/result.mp4`): Video with bounding boxes and player IDs
2. **Results JSON** (`results/tracking_results.json`): Detailed tracking data
3. **Results CSV** (`results/tracking_results.csv`): Frame-wise player positions
4. **Analysis Plots** (if `--show-plots` is used):
   - `track_positions.png`: Player movement visualization
   - `player_count_over_time.png`: Number of players over time
   - `tracking_summary.txt`: Text summary of results

## Technical Details

### Detection Module (`src/detector.py`)
- Uses YOLOv11 for player detection
- Configurable confidence and IoU thresholds
- Extracts features for re-identification

### Tracking Module (`src/tracker.py`)
- Implements multi-object tracking with re-identification
- Uses IoU and feature similarity for matching
- Maintains track history and handles occlusions

### Visualization Module (`src/visualizer.py`)
- Draws bounding boxes and player IDs
- Creates analysis plots and statistics
- Supports track history visualization

### Pipeline Module (`src/pipeline.py`)
- Orchestrates the entire processing pipeline
- Handles video I/O and result generation
- Provides comprehensive statistics

## Performance

- **Processing Speed**: ~5-15 FPS on CPU, ~20-30 FPS on GPU
- **Memory Usage**: ~2-4 GB RAM for 720p videos
- **Accuracy**: Depends on model quality and video conditions

## Troubleshooting

### Common Issues

1. **Model not found**:
   - Ensure the YOLOv11 model is in the `models/` directory
   - Check the model file path in the command

2. **Video not found**:
   - Verify the input video path is correct
   - Ensure the video file exists and is readable

3. **CUDA out of memory**:
   - Reduce batch size or use CPU processing
   - Close other GPU-intensive applications

4. **Import errors**:
   - Ensure all dependencies are installed: `pip install -r requirements.txt`
   - Check Python version (3.8+ required)

### Performance Optimization

- Use GPU acceleration if available
- Adjust confidence thresholds based on video quality
- Reduce video resolution for faster processing
- Use `--no-progress` flag to reduce overhead

## Methodology

### Detection Approach
1. **YOLOv11 Model**: Pre-trained model fine-tuned for player detection
2. **Confidence Filtering**: Removes low-confidence detections
3. **Non-Maximum Suppression**: Eliminates overlapping detections

### Tracking Approach
1. **IoU Matching**: Primary method for tracking across frames
2. **Feature Similarity**: Secondary method for re-identification
3. **Hungarian Algorithm**: Optimal assignment of detections to tracks
4. **Track Management**: Handles track creation, updating, and deletion

### Re-Identification Approach
1. **Feature Extraction**: Histogram-based features from player regions
2. **Similarity Calculation**: Cosine similarity between feature vectors
3. **Threshold-based Matching**: Configurable similarity thresholds
4. **Track History**: Maintains feature history for robust matching

## Future Improvements

- **Advanced Re-ID Models**: Integration with dedicated re-identification models
- **Multi-Camera Support**: Tracking across multiple camera feeds
- **Real-time Processing**: Optimizations for live video streams
- **Web Interface**: GUI for easier interaction
- **Advanced Analytics**: Player behavior analysis and statistics

## License

This project is developed for the Liat.ai AI Intern assignment.

## Contact

For questions or issues, please contact:
- arshdeep@liat.ai
- rishit@liat.ai

## Acknowledgments

- Ultralytics for YOLOv11 implementation
- OpenCV for computer vision utilities
- PyTorch for deep learning framework 