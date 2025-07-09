#!/usr/bin/env python3
"""
Player Re-Identification System
Main CLI script for processing sports videos

Usage:
    python main.py --input video.mp4 --model model.pt --output output.mp4
"""

import argparse
import os
import sys
import time
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from src.detector import PlayerDetector
from src.tracker import PlayerTracker
from src.visualizer import VideoVisualizer, ResultsVisualizer
from src.pipeline import PlayerReIDPipeline


def main():
    """Main function for CLI interface"""
    parser = argparse.ArgumentParser(
        description="Player Re-Identification System for Sports Videos",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python main.py --input data/15sec_input_720p.mp4 --model models/yolov11.pt --output output/result.mp4
    python main.py --input video.mp4 --model model.pt --output result.mp4 --conf 0.6 --show-plots
        """
    )
    
    # Required arguments
    parser.add_argument("--input", "-i", required=True,
                       help="Input video file path")
    parser.add_argument("--model", "-m", required=True,
                       help="YOLOv11 model file path")
    parser.add_argument("--output", "-o", required=True,
                       help="Output video file path")
    
    # Optional arguments
    parser.add_argument("--conf", type=float, default=0.5,
                       help="Detection confidence threshold (default: 0.5)")
    parser.add_argument("--iou", type=float, default=0.45,
                       help="Detection IoU threshold (default: 0.45)")
    parser.add_argument("--track-iou", type=float, default=0.3,
                       help="Tracking IoU threshold (default: 0.3)")
    parser.add_argument("--feature-threshold", type=float, default=0.7,
                       help="Feature similarity threshold (default: 0.7)")
    parser.add_argument("--max-missed", type=int, default=30,
                       help="Maximum frames to keep inactive tracks (default: 30)")
    parser.add_argument("--results", "-r", default="results",
                       help="Results output directory (default: results)")
    parser.add_argument("--show-plots", action="store_true",
                       help="Generate analysis plots")
    parser.add_argument("--no-progress", action="store_true",
                       help="Disable progress bar")
    
    args = parser.parse_args()
    
    # Validate input files
    if not os.path.exists(args.input):
        print(f"Error: Input video file not found: {args.input}")
        sys.exit(1)
    
    if not os.path.exists(args.model):
        print(f"Error: Model file not found: {args.model}")
        sys.exit(1)
    
    # Create output directories
    output_dir = Path(args.output).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results_dir = Path(args.results)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup output paths
    output_video_path = args.output
    output_results_path = str(results_dir / "tracking_results")
    
    print("=== Player Re-Identification System ===")
    print(f"Input video: {args.input}")
    print(f"Model: {args.model}")
    print(f"Output video: {output_video_path}")
    print(f"Results directory: {results_dir}")
    print(f"Detection confidence: {args.conf}")
    print(f"Detection IoU: {args.iou}")
    print(f"Tracking IoU: {args.track_iou}")
    print(f"Feature threshold: {args.feature_threshold}")
    print(f"Max missed frames: {args.max_missed}")
    print()
    
    try:
        # Initialize pipeline
        print("Initializing pipeline...")
        pipeline = PlayerReIDPipeline(
            model_path=args.model,
            conf_threshold=args.conf,
            iou_threshold=args.iou,
            track_iou_threshold=args.track_iou,
            feature_threshold=args.feature_threshold,
            max_missed_frames=args.max_missed
        )
        
        # Process video
        print("Starting video processing...")
        start_time = time.time()
        
        summary = pipeline.process_video(
            input_video_path=args.input,
            output_video_path=output_video_path,
            output_results_path=output_results_path,
            show_progress=not args.no_progress
        )
        
        total_time = time.time() - start_time
        
        # Print results
        print("\n=== Processing Results ===")
        print(f"Total processing time: {total_time:.2f} seconds")
        print(f"Video processing speed: {summary['fps_processing']:.2f} FPS")
        print(f"Total frames processed: {summary['total_frames']}")
        print(f"Total players tracked: {summary['total_players_tracked']}")
        print(f"Video resolution: {summary['video_resolution']}")
        print(f"Video FPS: {summary['video_fps']}")
        
        # Generate analysis plots if requested
        if args.show_plots:
            print("\nGenerating analysis plots...")
            pipeline.create_analysis_plots(str(results_dir))
        
        # Get and print tracking statistics
        stats = pipeline.get_tracking_statistics()
        if stats:
            print("\n=== Tracking Statistics ===")
            print(f"Total tracks: {stats['total_tracks']}")
            print(f"Total frames: {stats['total_frames']}")
            
            if stats['track_lengths']:
                avg_track_length = sum(stats['track_lengths'].values()) / len(stats['track_lengths'])
                print(f"Average track length: {avg_track_length:.1f} frames")
                
                longest_track = max(stats['track_lengths'].values())
                print(f"Longest track: {longest_track} frames")
        
        print(f"\nResults saved to: {results_dir}")
        print("Processing completed successfully!")
        
    except Exception as e:
        print(f"Error during processing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main() 