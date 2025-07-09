"""
Main Pipeline Module
Orchestrates the entire player re-identification process
"""

import cv2
import numpy as np
import json
import csv
import os
from typing import List, Dict, Any, Optional
from tqdm import tqdm
import time

from .detector import PlayerDetector
from .tracker import PlayerTracker
from .visualizer import VideoVisualizer, ResultsVisualizer
import matplotlib.pyplot as plt


class PlayerReIDPipeline:
    """Main pipeline for player re-identification"""
    
    def __init__(self, 
                 model_path: str,
                 conf_threshold: float = 0.5,
                 iou_threshold: float = 0.45,
                 track_iou_threshold: float = 0.3,
                 feature_threshold: float = 0.7,
                 max_missed_frames: int = 30):
        """
        Initialize the pipeline
        
        Args:
            model_path: Path to YOLOv11 model
            conf_threshold: Detection confidence threshold
            iou_threshold: Detection IoU threshold
            track_iou_threshold: Tracking IoU threshold
            feature_threshold: Feature similarity threshold
            max_missed_frames: Maximum frames to keep inactive tracks
        """
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.track_iou_threshold = track_iou_threshold
        self.feature_threshold = feature_threshold
        self.max_missed_frames = max_missed_frames
        
        # Initialize components
        self.detector = PlayerDetector(model_path, conf_threshold, iou_threshold)
        self.tracker = PlayerTracker(track_iou_threshold, feature_threshold, max_missed_frames)
        self.visualizer = VideoVisualizer()
        self.results_visualizer = ResultsVisualizer()
        
        # Results storage
        self.frame_results = []
        self.track_data = {}
        
    def process_video(self, 
                     input_video_path: str, 
                     output_video_path: str,
                     output_results_path: str,
                     show_progress: bool = True) -> Dict[str, Any]:
        """
        Process a video file for player re-identification
        
        Args:
            input_video_path: Path to input video
            output_video_path: Path to output video with visualizations
            output_results_path: Path to save results (CSV/JSON)
            show_progress: Whether to show progress bar
            
        Returns:
            Dictionary with processing results and statistics
        """
        print(f"Processing video: {input_video_path}")
        
        # Open video
        cap = cv2.VideoCapture(input_video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {input_video_path}")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Video properties: {width}x{height}, {fps} FPS, {total_frames} frames")
        
        # Setup output video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
        
        # Processing loop
        frame_count = 0
        start_time = time.time()
        
        progress_bar = tqdm(total=total_frames, desc="Processing frames") if show_progress else None
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Detect players
            detections = self.detector.detect_players(frame)
            
            # Track players
            tracked_players = self.tracker.update(detections, frame)
            
            # Store results
            frame_result = {
                'frame_number': frame_count,
                'timestamp': frame_count / fps,
                'detections': detections,
                'tracked_players': tracked_players
            }
            self.frame_results.append(frame_result)
            
            # Update track data
            for player in tracked_players:
                track_id = player['track_id']
                if track_id not in self.track_data:
                    self.track_data[track_id] = []
                self.track_data[track_id].append(player['bbox'])
            
            # Visualize results
            output_frame = self.visualizer.draw_tracks(frame, tracked_players)
            output_frame = self.visualizer.add_frame_info(
                output_frame, frame_count, len(tracked_players), fps
            )
            
            # Write output frame
            out.write(output_frame)
            
            if progress_bar:
                progress_bar.update(1)
        
        # Cleanup
        cap.release()
        out.release()
        if progress_bar:
            progress_bar.close()
        
        processing_time = time.time() - start_time
        
        # Save results
        self._save_results(output_results_path)
        
        # Create summary
        summary = {
            'input_video': input_video_path,
            'output_video': output_video_path,
            'output_results': output_results_path,
            'total_frames': frame_count,
            'processing_time': processing_time,
            'fps_processing': frame_count / processing_time,
            'total_players_tracked': len(self.track_data),
            'video_fps': fps,
            'video_resolution': f"{width}x{height}"
        }
        
        print(f"Processing completed in {processing_time:.2f} seconds")
        print(f"Average processing speed: {frame_count / processing_time:.2f} FPS")
        print(f"Total players tracked: {len(self.track_data)}")
        
        return summary
    
    def _save_results(self, output_path: str):
        """Save results to CSV and JSON files"""
        base_path = os.path.splitext(output_path)[0]
        
        # Save as JSON
        json_path = f"{base_path}.json"
        with open(json_path, 'w') as f:
            json.dump({
                'frame_results': self.frame_results,
                'track_data': self.track_data
            }, f, indent=2)
        
        # Save as CSV
        csv_path = f"{base_path}.csv"
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['frame_number', 'timestamp', 'track_id', 'x1', 'y1', 'x2', 'y2', 'confidence'])
            
            for frame_result in self.frame_results:
                frame_number = frame_result['frame_number']
                timestamp = frame_result['timestamp']
                
                for player in frame_result['tracked_players']:
                    track_id = player['track_id']
                    bbox = player['bbox']
                    confidence = player.get('confidence', 1.0)
                    
                    writer.writerow([
                        frame_number, timestamp, track_id,
                        bbox[0], bbox[1], bbox[2], bbox[3], confidence
                    ])
        
        print(f"Results saved to {json_path} and {csv_path}")
    
    def create_analysis_plots(self, output_dir: str):
        """Create analysis plots and save them"""
        if not self.track_data:
            print("No track data available for analysis")
            return
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Get video dimensions from first frame result
        if self.frame_results:
            # Estimate video dimensions from bounding boxes
            all_bboxes = []
            for frame_result in self.frame_results:
                for player in frame_result['tracked_players']:
                    all_bboxes.append(player['bbox'])
            
            if all_bboxes:
                max_x = max(bbox[2] for bbox in all_bboxes)
                max_y = max(bbox[3] for bbox in all_bboxes)
                video_width = max_x
                video_height = max_y
            else:
                video_width, video_height = 1920, 1080  # Default
        else:
            video_width, video_height = 1920, 1080  # Default
        
        # Create track positions plot
        fig1 = self.results_visualizer.plot_track_positions(
            self.track_data, video_width, video_height
        )
        fig1.savefig(os.path.join(output_dir, 'track_positions.png'), dpi=300, bbox_inches='tight')
        plt.close(fig1)
        
        # Create player count plot
        fig2 = self.results_visualizer.plot_player_count_over_time(self.frame_results)
        fig2.savefig(os.path.join(output_dir, 'player_count_over_time.png'), dpi=300, bbox_inches='tight')
        plt.close(fig2)
        
        # Create summary text
        summary_text = self.results_visualizer.create_tracking_summary(
            self.track_data, len(self.frame_results)
        )
        
        with open(os.path.join(output_dir, 'tracking_summary.txt'), 'w') as f:
            f.write(summary_text)
        
        print(f"Analysis plots saved to {output_dir}")
    
    def get_tracking_statistics(self) -> Dict[str, Any]:
        """Get comprehensive tracking statistics"""
        if not self.track_data:
            return {}
        
        stats = {
            'total_tracks': len(self.track_data),
            'total_frames': len(self.frame_results),
            'track_lengths': {},
            'track_durations': {},
            'average_positions': {}
        }
        
        for track_id, bboxes in self.track_data.items():
            if len(bboxes) == 0:
                continue
            
            # Track length
            stats['track_lengths'][track_id] = len(bboxes)
            
            # Track duration (assuming 30 FPS)
            stats['track_durations'][track_id] = len(bboxes) / 30.0
            
            # Average position
            centers_x = [(bbox[0] + bbox[2]) / 2 for bbox in bboxes]
            centers_y = [(bbox[1] + bbox[3]) / 2 for bbox in bboxes]
            avg_x = sum(centers_x) / len(centers_x)
            avg_y = sum(centers_y) / len(centers_y)
            stats['average_positions'][track_id] = (avg_x, avg_y)
        
        return stats 